# Find mask threshold
# python -m iw3.cli -i ./images/ -o ./export_images --yes --depth-model Any_V2_B --export-disparity
# python -m iw3.training.sbs.find_mask_threshold --rgb-dir ./export_images/rgb --depth-dir ./export_images/depth
#
# Visualize
# python -m iw3.training.sbs.find_mask_threshold --rgb-dir ./export_images/rgb --depth-dir ./export_images/depth -o tmp/mask_vis --threshold 0.15

import argparse
import os
from os import path
import torch
import torchvision.transforms.functional as TF
from iw3 import models  # noqa
from iw3.stereo_model_factory import create_stereo_model
from iw3.forward_warp import nonwarp_mask as forward_nonwarp_mask
from iw3.backward_warp import nonwarp_mask as backward_nonwarp_mask
from iw3.dilation import mask_closing
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import ImageLoader
from tqdm import tqdm


def iou(mask1, mask2):
    m1 = mask1.bool()
    m2 = mask2.bool()

    intersection = (m1 & m2).sum().float()
    union = (m1 | m2).sum().float()

    if union == 0:
        return 1.0
    return (intersection / union).item()


def bench(args, model, divergence, threshold):
    print(f"** threshold = {threshold}")

    rgb_files = ImageLoader.listdir(args.rgb_dir)
    depth_files = ImageLoader.listdir(args.depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{args.rgb_dir} is empty")

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "gray"})

    iou_sum = 0
    for rgb, depth in tqdm(zip(rgb_loader, depth_loader)):
        rgb = TF.to_tensor(rgb[0]).unsqueeze(0).cuda()
        depth = TF.to_tensor(depth[0]).unsqueeze(0).cuda()

        _, forward_mask = forward_nonwarp_mask(rgb, depth, divergence=divergence, convergence=0.5)
        _, backward_mask = backward_nonwarp_mask(model, rgb, depth, divergence=divergence, convergence=0.5, mapper="none")

        forward_mask = (forward_mask > 0.9).float()
        forward_mask = mask_closing(forward_mask)
        backward_mask = (backward_mask > threshold).float()
        # backward_mask = mask_closing(backward_mask)

        iou_sum += iou(forward_mask, backward_mask)

    print(f"IOU: {iou_sum/len(rgb_files)}")


def visualize(args, model, divergence=2.0, threshold=0.4):
    print(f"** threshold = {threshold}")

    rgb_files = ImageLoader.listdir(args.rgb_dir)
    depth_files = ImageLoader.listdir(args.depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{args.rgb_dir} is empty")

    os.makedirs(args.output_dir, exist_ok=True)

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "gray"})

    i = 0
    for rgb, depth in tqdm(zip(rgb_loader, depth_loader)):
        rgb = TF.to_tensor(rgb[0]).unsqueeze(0).cuda()
        depth = TF.to_tensor(depth[0]).unsqueeze(0).cuda()

        _, forward_mask = forward_nonwarp_mask(rgb, depth, divergence=divergence, convergence=0.5)
        _, backward_mask = backward_nonwarp_mask(model, rgb, depth, divergence=divergence, convergence=0.5, mapper="none")

        forward_mask = (forward_mask > 0.9).float()
        forward_mask = mask_closing(forward_mask)
        backward_mask = (backward_mask > threshold).float()

        mask_pack = torch.cat([forward_mask, backward_mask, rgb.mean(dim=1, keepdim=True)], dim=1)

        i += 1
        TF.to_pil_image(mask_pack[0]).save(path.join(args.output_dir, f"{i}.png"))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rgb-dir", type=str, required=True, help="input rgb data dir")
    parser.add_argument("--depth-dir", type=str, required=True, help="input depth data dir")
    parser.add_argument("--output-dir", "-o", type=str, help="output dir for visualization")
    parser.add_argument("--threshold", type=float, default=0.15, help="threshold for visualization")
    parser.add_argument("--divergence", type=float, default=2.0, help="divergence right only = 1/2")

    args = parser.parse_args()

    mask_mlbw = create_stereo_model("mask_mlbw_l2", divergence=1, device_id=0)

    if args.output_dir:
        # visualize
        visualize(args, mask_mlbw, divergence=args.divergence, threshold=args.threshold)
    else:
        # find parameter
        for threshold in torch.arange(0.025, 0.5, 0.025):
            bench(args, mask_mlbw, divergence=args.divergence, threshold=threshold)


if __name__ == "__main__":
    main()
