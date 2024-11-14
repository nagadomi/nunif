# wip video stabilizer
#
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/ --border buffer --debug
# sample: https://github.com/user-attachments/assets/abded633-58ec-42d8-9510-0c7adf043326
import os
from os import path
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import nunif.utils.video as VU
import nunif.utils.superpoint as KU
from nunif.device import mps_is_available, xpu_is_available, create_device
from tqdm import tqdm
import scipy


SUPERPOINT_CONF = {
    "nms_radius": 4,
    "max_num_keypoints": None,
    "detection_threshold": 0.01,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
}


ANGLE_MAX_HARD = 90.0


def resize(x, size):
    B, C, H, W = x.shape
    if H < W:
        new_height = size
        new_width = int(W / (H / size))
    else:
        new_width = size
        new_height = int(H / (W / size))

    height_scale = H / new_height
    width_scale = W / new_width
    x = F.interpolate(x, (new_height, new_width), mode="bilinear", align_corners=False, antialias=False)

    return x, (height_scale + width_scale) * 0.5


def plot_keypoints(x, kp):
    img = TF.to_pil_image(x.cpu())
    gc = ImageDraw.Draw(img)
    for xy in kp:
        xx, yy = int(xy[0].item()), int(xy[1].item())
        gc.circle((xx, yy), radius=2, fill="red")

    return img


def gen_gaussian_kernel(kernel_size, device):
    sigma = kernel_size * 0.15 + 0.35
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=torch.float64, device=device)
    gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel.reshape(1, 1, -1)


def gen_savgol_kernel(kernel_size, device):
    kernel = scipy.signal.savgol_coeffs(kernel_size, polyorder=2)
    return torch.from_numpy(kernel).to(device).reshape(1, 1, -1)


def replication_pad1d_naive(x, padding, detach=False):
    assert x.ndim == 3 and len(padding) == 4
    left, right, top, bottom = padding

    detach_fn = lambda t: t.detach() if detach else t
    if left > 0:
        x = torch.cat((*((detach_fn(x[:, :, :1]),) * left), x), dim=2)
    elif left < 0:
        x = x[:, :, -left:]
    if right > 0:
        x = torch.cat((x, *((detach_fn(x[:, :, -1:]),) * right)), dim=2)
    elif right < 0:
        x = x[:, :, :right]

    return x.contiguous()


def smoothing(x, weight):
    kernel_size = weight.shape[2]
    padding = (kernel_size - 1) // 2
    x = replication_pad1d_naive(x, (padding, padding, 0, 0))
    x = F.conv1d(x, weight=weight, stride=1, padding=0)
    return x


def calc_scene_weight(mean_match_scores, device=None):
    # mean_match_scores: mean of best match keypoint's cosine similarity
    # when score < 0.5, it is highly likely that scene change has occurred
    # when score < 0.65, scene change possibly has occurred
    # when score > 0.75, it is probably safe range
    if torch.is_tensor(mean_match_scores):
        score = mean_match_scores
    else:
        score = torch.tensor(mean_match_scores, dtype=torch.float32, device=device)

    max_score = 0.75
    min_score = 0.5
    weight = ((score - min_score) / (max_score - min_score)).clamp(0, 1)
    weight[weight < 0.65] = weight[weight < 0.65] ** 2

    return weight


def video_config_callback(args, fps_hook=None):
    def callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        if fps_hook is not None:
            fps_hook(fps)
        return VU.VideoOutputConfig(
            fps=fps,
            options={"preset": args.preset, "crf": str(args.crf)}
        )
    return callback


def list_chunk(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def pass1(args, device):
    keypoint_model = KU.SuperPoint(**SUPERPOINT_CONF).load().to(device)
    mean_match_scores = []
    points1 = []
    points2 = []
    kp_seam = [None]
    center = [[0, 0]]
    resize_scale = [1.0]

    def keypoint_callback(x, pts):
        with torch.inference_mode(), torch.autocast(device_type=device.type):
            x, resize_scale[0] = resize(x, args.resolution)
            center[0] = [x.shape[3] // 2, x.shape[2] // 2]
            kp_batch = keypoint_model.infer(x)

        kp_batch.insert(0, kp_seam[0])
        kp_seam[0] = kp_batch[-1]

        for i in range(1, len(kp_batch)):
            kp1 = kp_batch[i - 1]
            kp2 = kp_batch[i]
            if kp1 is None or (kp1["keypoints"].shape[0] == 0 or kp2["keypoints"].shape[0] == 0):
                zero_points = torch.zeros((0, 2), dtype=x.dtype, device=torch.device("cpu"))
                mean_match_scores.append(0.0)
                points1.append(zero_points)
                points2.append(zero_points)
                continue

            index1, index2, match_score = KU.find_match_index(
                kp1, kp2,
                threshold=0.6, return_score_all=True)
            kp1 = kp1["keypoints"][index1]
            kp2 = kp2["keypoints"][index2]

            mean_match_scores.append(match_score.mean().item())
            points1.append(kp1.cpu())
            points2.append(kp2.cpu())

    fps_value = [0]

    def fps_hook(fps):
        fps_value[0] = fps

    keypoint_callback_pool = VU.FrameCallbackPool(
        keypoint_callback,
        require_pts=True,
        batch_size=args.batch_size,
        device=device,
        max_workers=0,  # must be sequential
    )

    VU.hook_frame(args.input, keypoint_callback_pool,
                  config_callback=video_config_callback(args, fps_hook),
                  title="pass 1/3")

    return points1, points2, mean_match_scores, center[0], resize_scale[0], fps_value[0]


def pack_points(batch1, batch2):
    fixed_size = max(max(pts.shape[0] for pts in batch1), max(pts.shape[0] for pts in batch2))
    batch1_fixed = []
    batch2_fixed = []
    batch_mask = []
    for pts1, pts2 in zip(batch1, batch2):
        assert pts1.shape[0] == pts2.shape[0]
        pack1 = torch.zeros((fixed_size, pts1.shape[1]), dtype=pts1.dtype, device=pts1.device)
        pack2 = torch.zeros((fixed_size, pts1.shape[1]), dtype=pts1.dtype, device=pts1.device)
        mask = torch.zeros((fixed_size, pts1.shape[1]), dtype=torch.bool, device=pts1.device)

        pack1[:pts1.shape[0]] = pts1
        pack2[:pts2.shape[0]] = pts2
        mask[:pts1.shape[0]] = True

        batch1_fixed.append(pack1)
        batch2_fixed.append(pack2)
        batch_mask.append(mask)

    return torch.stack(batch1_fixed), torch.stack(batch2_fixed), torch.stack(batch_mask)


def pass2(points1, points2, center, resize_scale, args, device):
    if len(points1) == 0:
        return []

    transforms = []

    points1, points2, masks = pack_points(points1, points2)
    batch_size = args.batch_size * 32

    pbar = tqdm(total=len(points1), ncols=80, desc="pass 2/3")
    for kp1, kp2, mask in zip(points1.split(batch_size), points2.split(batch_size), masks.split(batch_size)):
        kp1, kp2, mask = kp1.to(device), kp2.to(device), mask.to(device)
        center_batch = torch.tensor(center, dtype=torch.float32, device=device).view(1, 2).expand(kp1.shape[0], 1, 2)
        shift, scale, angle, center_batch = KU.find_rigid_transform(
            kp1, kp2, center=center_batch, mask=mask,
            iteration=args.iteration, sigma=2.0,
            disable_scale=True)
        for i in range(kp1.shape[0]):
            transforms.append((shift[i].tolist(), scale[i].item(), angle[i].item(), center, resize_scale))
            pbar.update(1)
    pbar.close()
    return transforms


def pass3(transforms, mean_match_scores, kernel_size, args, device):
    if path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_dir = args.output
        output_file_path = path.join(output_dir, path.splitext(path.basename(args.input))[0] + "_vidstab.mp4")
    else:
        output_dir = path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = args.output

    shift_x = torch.tensor([rec[0][0] for rec in transforms], dtype=torch.float64, device=device)
    shift_y = torch.tensor([rec[0][1] for rec in transforms], dtype=torch.float64, device=device)
    angle = torch.tensor([rec[2] for rec in transforms], dtype=torch.float64, device=device)
    # limit angle
    angle = angle.clamp(-ANGLE_MAX_HARD, ANGLE_MAX_HARD)

    scene_weight = calc_scene_weight(mean_match_scores, device=device) * args.strength
    shift_x = shift_x * scene_weight  # + 0 * (1 - scene_weight)
    shift_y = shift_y * scene_weight
    angle = angle * scene_weight

    shift_x = shift_x.cumsum(dim=0).reshape(1, 1, -1)
    shift_y = shift_y.cumsum(dim=0).reshape(1, 1, -1)
    angle = angle.cumsum(dim=0).reshape(1, 1, -1)

    if args.filter == "gaussian":
        kernel = gen_gaussian_kernel(kernel_size, device)
    else:
        kernel = gen_savgol_kernel(kernel_size, device)

    shift_x_smooth = smoothing(shift_x, kernel)
    shift_y_smooth = smoothing(shift_y, kernel)
    angle_smooth = smoothing(angle, kernel)

    shift_x_fix = (shift_x_smooth - shift_x).flatten()
    shift_y_fix = (shift_y_smooth - shift_y).flatten()
    angle_fix = (angle_smooth - angle).flatten()

    def test_callback(frame):
        if frame is None:
            return None
        im = frame.to_image()
        x = TF.to_tensor(im).to(device)

        if args.border == "expand":
            padding = int(max(x.shape[1], x.shape[2]) * args.padding)
            x = F.pad(x, (padding,) * 4, mode="constant", value=0)
        elif args.border == "crop":
            padding = int(max(x.shape[1], x.shape[2]) * args.padding)
            x = F.pad(x, (-padding,) * 4)

        if args.debug:
            z = torch.cat([x, x], dim=2)
            return VU.to_frame(z)
        else:
            return VU.to_frame(x)

    index = [0]
    buffer = [None]

    def stabilizer_callback(x):
        B = x.shape[0]
        i = index[0]
        index[0] += x.shape[0]

        # assume all values are the same
        center = transforms[i][3]
        resize_scale = transforms[i][4]

        if args.border == "buffer":
            padding = int(max(x.shape[2], x.shape[3]) * args.padding)
            x_input = F.pad(x, (padding,) * 4, mode="constant", value=torch.nan)
            center = [center[0] + padding, center[1] + padding]
            padding_mode = "reflection"
        elif args.border == "expand":
            padding = int(max(x.shape[2], x.shape[3]) * args.padding)
            x_input = F.pad(x, (padding,) * 4, mode="constant", value=0)
            center = [center[0] + padding, center[1] + padding]
            padding_mode = "zeros"
        elif args.border == "crop":
            padding = 0
            x_input = x
            padding_mode = "reflection"
        else:
            padding = 0
            x_input = x
            padding_mode = args.border

        shifts = torch.tensor([[shift_x_fix[i + j].item() * resize_scale,
                                shift_y_fix[i + j].item() * resize_scale] for j in range(B)],
                              dtype=x.dtype, device=x.device)
        centers = torch.tensor([center for _ in range(B)], dtype=x.dtype, device=x.device)
        angles = torch.tensor([angle_fix[i + j] for j in range(B)],
                              dtype=x.dtype, device=x.device)
        scales = torch.ones((B,), dtype=x.dtype, device=x.device)

        z = KU.apply_rigid_transform(x_input, shifts, scales, angles, centers, padding_mode=padding_mode)

        if args.border == "buffer":
            z = F.pad(z, (-padding,) * 4)
            # Update EMA frame buffer
            for j in range(z.shape[0]):
                if buffer[0] is None or scene_weight[i + j] < 0.01:
                    # reset buffer
                    buffer[0] = x[j].clone()
                mask = torch.isnan(z[j])
                mask_not = torch.logical_not(mask)
                buffer[0][mask_not] = buffer[0][mask_not] * args.buffer_decay + z[j][mask_not] * (1.0 - args.buffer_decay)
                z[j][mask] = buffer[0][mask]
            z.clamp_(0, 1)
        elif args.border == "crop":
            padding = int(max(z.shape[2], z.shape[3]) * args.padding)
            z = F.pad(z, (-padding,) * 4).clamp_(0, 1)
        else:
            z = z.clamp(0, 1)

        if args.debug:
            if args.border == "expand":
                x = x_input
            elif args.border == "crop":
                x = F.pad(x_input, (-padding,) * 4)
            z = torch.cat([x, z], dim=3)

        return z

    stabilizer_callback_pool = VU.FrameCallbackPool(
        stabilizer_callback,
        batch_size=args.batch_size,
        device=device,
        max_workers=0,
    )
    VU.process_video(args.input, output_file_path,
                     stabilizer_callback_pool,
                     config_callback=video_config_callback(args),
                     test_callback=test_callback,
                     title="pass 3/3")


def main():
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--gpu", "-g", type=int, default=default_gpu,
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--smoothing", type=float, default=2.0, help="seconds to smoothing")
    parser.add_argument("--filter", type=str, default="savgol", choices=["gaussian", "savgol"], help="smoothing filter")

    parser.add_argument("--border", type=str, choices=["zeros", "border", "reflection", "buffer", "expand", "crop"],
                        default="zeros", help="border padding mode")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="pre-padding ratio for --border=buffer|expand|crop")
    parser.add_argument("--buffer-decay", type=float, default=0.75,
                        help="buffer decay factor for --border=buffer")

    parser.add_argument("--debug", action="store_true", help="debug output original+stabilized")
    parser.add_argument("--strength", type=float, default=1.0, help="influence 0.0-1.0")
    parser.add_argument("--resolution", type=int, default=320, help="resolution to perform processing")
    parser.add_argument("--iteration", type=int, default=50, help="iteration count of frame transform optimization")
    parser.add_argument("--max-fps", type=float, default=30.0,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo"],
                        help="encoder preset option for video")
    parser.add_argument("--crf", type=int, default=16,
                        help="constant quality value for video. smaller value is higher quality")

    args = parser.parse_args()
    assert 0 <= args.strength <= 1.0 and args.strength

    device = create_device(args.gpu)

    # detect keypoints and matching
    points1, points2, mean_match_scores, center, resize_scale, fps = pass1(args=args, device=device)

    # select smoothing kernel size
    kernel_size = int(args.smoothing * float(fps))
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # calculate optical flow (rigid transform)
    transforms = pass2(points1, points2, center, resize_scale, args=args, device=device)

    assert len(transforms) == len(mean_match_scores)

    # stabilize
    pass3(transforms, mean_match_scores, kernel_size, args=args, device=device)


if __name__ == "__main__":
    main()
