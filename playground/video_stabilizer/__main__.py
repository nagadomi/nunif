# wip video stabilizer
#
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/
# python -m playground.video_stabilizer -i ./tmp/test_videos/bottle_vidstab.mp4 -o ./tmp/vidstab/ --debug
# https://github.com/user-attachments/assets/7cdff090-7293-4054-a265-ca355883f2d6
# TODO: batch processing (find_rigid_transform, apply_rigid_transform)
#       padding before transform and unpadding after transform
import os
from os import path
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import threading
import gzip
import json
import nunif.utils.video as VU
import nunif.utils.superpoint as KU
from nunif.device import device_is_cuda
import time
from tqdm import tqdm


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


def smoothing(x, weight, center_value=0.0):
    kernel_size = weight.shape[2]
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
            options={"preset": "medium", "crf": "16"}
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
                zero_points = torch.zeros((0, 2), dtype=x.dtype, device=device)
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
            points1.append(kp1)
            points2.append(kp2)

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


def pack_descriptors(batch1, batch2):
    fixed_size = max(max(d1.shape[0] for d1 in batch1), max(d2.shape[0] for d2 in batch2))

    fixed_batch1 = []
    fixed_batch2 = []
    for d1 in batch1:
        if fixed_size != d1.shape[0]:
            pack = torch.zeros((fixed_size, d1.shape[1]), dtype=d1.dtype, device=d1.device)
            pack[:d1.shape[0]] = d1
            d1 = pack
    for d2 in batch2:
        if fixed_size != d2.shape[0]:
            pack = torch.zeros((fixed_size, d2.shape[1]), dtype=d2.dtype, device=d2.device)
            pack[:d2.shape[0]] = d2
            d2 = pack

        fixed_batch2.append(d2)

    return torch.stack(fixed_batch1), torch.stack(fixed_batch2)


def pass2(points1, points2, center, resize_scale, args, device):
    if len(points1) == 0:
        return []

    default_keypoint_rec = ([0.0, 0.0], 1.0, 0.0, center, resize_scale)
    transforms = []

    for kp1, kp2 in tqdm(zip(points1, points2), total=len(points1), ncols=80, desc="pass 2/3"):
        if kp1.shape[0] == 0 or kp2.shape[0] == 0:
            transforms.append(default_keypoint_rec)
            continue

        shift, scale, angle, center = KU.find_rigid_transform(kp1, kp2, center=center, sigma=2.0,
                                                              disable_scale=True)
        # fix input size scale
        transforms.append((shift, scale, angle, center, resize_scale))

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

    weight = calc_scene_weight(mean_match_scores, device=device) * args.strength
    shift_x = shift_x * weight  # + 0 * (1 - weight)
    shift_y = shift_y * weight
    angle = angle * weight

    shift_x = shift_x.cumsum(dim=0).reshape(1, 1, -1)
    shift_y = shift_y.cumsum(dim=0).reshape(1, 1, -1)
    angle = angle.cumsum(dim=0).reshape(1, 1, -1)

    gaussian_kernel = gen_gaussian_kernel(kernel_size, device)
    shift_x_smooth = smoothing(shift_x, gaussian_kernel)
    shift_y_smooth = smoothing(shift_y, gaussian_kernel)
    angle_smooth = smoothing(angle, gaussian_kernel)

    shift_x_fix = (shift_x_smooth - shift_x).flatten()
    shift_y_fix = (shift_y_smooth - shift_y).flatten()
    angle_fix = (angle_smooth - angle).flatten()

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
        x = VU.to_tensor(frame, device=device)

        center = transforms[i][3]
        resize_scale = transforms[i][4]
        z = KU.apply_rigid_transform(
            x,
            shift=[shift_x_fix[i].item() * resize_scale, shift_y_fix[i].item() * resize_scale],
            scale=1.0,
            angle=angle_fix[i].item(),
            center=[center[0] * resize_scale, center[1] * resize_scale],
            padding_mode=args.border,
        )
        index[0] += 1

        if args.debug:
            z = torch.cat([x, z], dim=2)

        return VU.to_frame(z)

    VU.process_video(args.input, output_file_path,
                     stabilizer_callback,
                     config_callback=video_config_callback(args),
                     test_callback=test_callback,
                     title="pass 3/3")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--max-fps", type=float, default=30.0, help="max fps")
    parser.add_argument("--smoothing", type=float, default=2.0, help="seconds to smoothing")
    parser.add_argument("--border", type=str, choices=["zeros", "border", "reflection"],
                        default="zeros", help="border padding mode")
    parser.add_argument("--debug", action="store_true", help="debug output original+stabilized")
    parser.add_argument("--strength", type=float, default=1.0, help="influence 0.0-1.0")
    parser.add_argument("--no-cache", action="store_true", help="do not use first pass cache")
    parser.add_argument("--resolution", type=int, default=320, help="resolution to perform processing")

    args = parser.parse_args()
    assert 0 <= args.strength <= 1.0 and args.strength

    device = torch.device("cuda")

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
