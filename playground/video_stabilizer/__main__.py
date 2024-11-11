# wip video stabilizer
#
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/ --debug
# https://github.com/user-attachments/assets/7cdff090-7293-4054-a265-ca355883f2d6
# TODO: correctly calculate the mean of angle. for example, mean([359, 0, 1]) should be 0.
#       padding before transform and unpadding after transform
#       batch processing (find_match_index, find_rigid_transform, apply_rigid_transform)
import os
from os import path
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import gzip
import json
import nunif.utils.video as VU
import nunif.utils.superpoint as KU


KEYPOINT_PROCESS_SIZE = 320

SUPERPOINT_CONF = {
    "nms_radius": 4,
    "max_num_keypoints": None,
    "detection_threshold": 0.01,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
}


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


def smoothing(x, weight, kernel_size, center_value=0.0):
    padding = (kernel_size - 1) // 2
    x = F.pad(x, (padding, padding, 0, 0), mode="constant", value=center_value)
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
    weight = weight ** 2

    # score  = [1.0000, 0.9000, 0.8000, 0.7500, 0.7000, 0.6500, 0.6000, 0.5000, 0.4000]
    # weight = [1.0000, 1.0000, 1.0000, 1.0000, 0.6400, 0.3600, 0.1600, 0.0000, 0.0000]

    return weight


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--max-fps", type=float, default=30.0, help="max fps")
    parser.add_argument("--kernel-size", type=int, default=61, help="smoothing kernel size")
    parser.add_argument("--border", type=str, choices=["zeros", "border", "reflection"],
                        default="zeros", help="border padding mode")
    parser.add_argument("--debug", action="store_true", help="debug output original+stabilized")
    args = parser.parse_args()
    assert args.kernel_size % 2 == 1

    device = torch.device("cuda")

    keypoint_model = KU.SuperPoint(**SUPERPOINT_CONF).load().to(device)

    if path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_dir = args.output
        output_file_path = path.join(output_dir, path.splitext(path.basename(args.input))[0] + "_vidstab.mp4")
    else:
        os.makedirs(path.dirname(args.output), exist_ok=True)
        output_dir = path.dirname(args.output)
        output_file_path = args.output

    transforms_file_path = path.join(output_dir, path.basename(args.input) + f"_{args.max_fps}.transforms.gz")
    score_file_path = path.join(output_dir, path.basename(args.input) + f"_{args.max_fps}.scores.gz")
    transforms = []

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        return VU.VideoOutputConfig(
            fps=fps,
            options={"preset": "medium", "crf": "16"}
        )

    kp_seam = [None]
    transforms = []
    mean_match_scores = []

    def keypoint_callback(x, pts):
        x, resize_scale = resize(x, KEYPOINT_PROCESS_SIZE)
        center = [x.shape[3] // 2, x.shape[2] // 2]
        with torch.inference_mode(), torch.autocast(device_type=device.type):
            kp_batch = keypoint_model.infer(x)

        kp_batch.insert(0, kp_seam[0])
        kp_seam[0] = kp_batch[-1]
        default_keypoint_rec = ([0.0, 0.0], 1.0, 0.0, center, resize_scale)
        for i in range(1, len(kp_batch)):
            if kp_batch[i - 1] is None:
                mean_match_scores.append(0.0)
                transforms.append(default_keypoint_rec)
                continue

            if kp_batch[i - 1]["keypoints"].shape[0] == 0 or kp_batch[i]["keypoints"].shape[0] == 0:
                mean_match_scores.append(0.0)
                transforms.append(default_keypoint_rec)
                continue

            index1, index2, match_score = KU.find_match_index(kp_batch[i - 1], kp_batch[i], threshold=0.5, return_score_all=True)

            mean_match_scores.append(match_score.mean().item())
            kp1 = kp_batch[i - 1]["keypoints"][index1]
            kp2 = kp_batch[i]["keypoints"][index2]

            if kp1.shape[0] == 0:
                transforms.append(default_keypoint_rec)
                continue

            # plot_transforms(x[i - 1], kp2).save(path.join(output_dir, f"debug_keypoint_{pts[i - 1]}.png"))

            shift, scale, angle, center = KU.find_rigid_transform(kp1, kp2, center=center, sigma=2.0,
                                                                  disable_scale=True)
            # fix input size scale
            transforms.append((shift, scale, angle, center, resize_scale))

    if args.debug or not path.exists(transforms_file_path):
        keypoint_callback_pool = VU.FrameCallbackPool(
            keypoint_callback,
            require_pts=True,
            batch_size=args.batch_size,
            device=device,
            max_workers=0)

        VU.hook_frame(args.input, keypoint_callback_pool,
                      config_callback=config_callback,
                      title="Tracking")
        with gzip.open(transforms_file_path, "wt") as f:
            f.write(json.dumps(transforms))
        with gzip.open(score_file_path, "wt") as f:
            f.write(json.dumps(mean_match_scores))

    with gzip.open(transforms_file_path, "rt") as f:
        transforms = json.load(f)
    with gzip.open(score_file_path, "rt") as f:
        mean_match_scores = json.load(f)

    # stabilize

    assert len(transforms) == len(mean_match_scores)

    shift_x = torch.tensor([rec[0][0] for rec in transforms], dtype=torch.float64, device=device)
    shift_y = torch.tensor([rec[0][1] for rec in transforms], dtype=torch.float64, device=device)
    rotate = torch.tensor([rec[2] for rec in transforms], dtype=torch.float64, device=device)

    # TODO: fix angle
    # adaptive_weight = calc_scene_weight(mean_match_scores, device=device)
    # shift_x = shift_x * adaptive_weight # + 0 * (1 - adaptive_weight)
    # shift_y = shift_y * adaptive_weight # + 0 * (1 - adaptive_weight)

    shift_x = shift_x.cumsum(dim=0).reshape(1, 1, -1)
    shift_y = shift_y.cumsum(dim=0).reshape(1, 1, -1)
    rotate = rotate.cumsum(dim=0).reshape(1, 1, -1)

    gaussian_kernel = gen_gaussian_kernel(args.kernel_size, device)
    shift_x_smooth = smoothing(shift_x, gaussian_kernel, args.kernel_size)
    shift_y_smooth = smoothing(shift_y, gaussian_kernel, args.kernel_size)
    rotate_smooth = smoothing(rotate, gaussian_kernel, args.kernel_size)

    shift_x_fix = (shift_x_smooth - shift_x).flatten()
    shift_y_fix = (shift_y_smooth - shift_y).flatten()
    rotate_fix = (rotate_smooth - rotate).flatten()

    def test_callback(frame):
        if frame is None:
            return None
        im = frame.to_image()
        x = TF.to_tensor(im).to(device)
        if args.debug:
            z = torch.cat([x, x], dim=2)
            return VU.to_frame(z)
        else:
            return VU.to_frame(x)

    index = [0]

    def stabilizer_callback(frame):
        if frame is None:
            return None
        i = index[0]
        if i >= len(transforms):
            return None

        im = frame.to_image()
        x = TF.to_tensor(im).to(device)

        center = transforms[i][3]
        resize_scale = transforms[i][4]
        z = KU.apply_rigid_transform(
            x,
            shift=[shift_x_fix[i].item() * resize_scale, shift_y_fix[i].item() * resize_scale],
            scale=1.0,
            angle=rotate_fix[i].item(),
            center=[center[0] * resize_scale, center[1] * resize_scale],
            padding_mode=args.border,
        )
        index[0] += 1

        if args.debug:
            z = torch.cat([x, z], dim=2)
            return VU.to_frame(z)
        else:
            return VU.to_frame(z)

    VU.process_video(args.input, output_file_path,
                     stabilizer_callback,
                     config_callback=config_callback,
                     test_callback=test_callback,
                     title="Stabilizing")


if __name__ == "__main__":
    main()
