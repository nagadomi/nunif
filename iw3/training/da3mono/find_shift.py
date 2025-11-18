import argparse
import os
from os import path
import torch
import torch.nn.functional as F
from iw3.base_depth_model import BaseDepthModel


def load_files(input_dir):
    files = []
    for fn in os.listdir(input_dir):
        if fn.endswith("_DA3.png"):
            fn_da2 = fn.replace("_DA3.png", "_DA2.png")
            files.append((path.join(input_dir, fn), path.join(input_dir, fn_da2)))

    return sorted(files)


def normalize(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())


def convert_to_disparity(depth, shift_ratio):
    depth_range = depth.max() - depth.min()
    return 1.0 / (depth + depth_range * shift_ratio)


def psnr(input, target):
    mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
    return 10 * torch.log10(1.0 / (mse + 1.0e-6))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input dir")
    args = parser.parse_args()

    for shift_ratio in torch.arange(0, 0.5, 0.02):
        shift_ratio = shift_ratio.item()
        sum_psnr = 0.0
        files = load_files(args.input)
        for da3, da2 in files:
            da3, _ = BaseDepthModel.load_depth(da3)
            da2, _ = BaseDepthModel.load_depth(da2)
            da2 = normalize(da2)
            da3 = normalize(convert_to_disparity(da3, shift_ratio=shift_ratio))

            sum_psnr += psnr(da3, da2)

        print(shift_ratio, sum_psnr / len(files))


if __name__ == "__main__":
    main()
