import argparse
import os
from os import path
import torch
import torch.nn.functional as F
import iw3.models  # noqa
from iw3.base_depth_model import BaseDepthModel
from nunif.models import load_model
from tqdm import tqdm


def load_files(input_dir):
    files = []
    for fn in os.listdir(input_dir):
        if fn.endswith("_DA3.png"):
            fn_da2 = fn.replace("_DA3.png", "_DA2.png")
            files.append((path.join(input_dir, fn), path.join(input_dir, fn_da2)))

    return sorted(files)


def convert_to_disparity(depth):
    if False:
        depth_range = depth.max() - depth.min()
        return 1.0 / (depth + depth_range * 0.1)
    else:
        return 1.0 / (depth + 0.35)


def normalize(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())


def psnr(input, target):
    mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
    return 10 * torch.log10(1.0 / (mse + 1.0e-6))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input dir")
    parser.add_argument("--model", type=str, required=True, help="model file")
    args = parser.parse_args()

    model = load_model(args.model)[0].eval().cuda()
    files = load_files(args.input)
    model_psnr = 0
    math_psnr = 0
    for da3, da2 in tqdm(files, ncols=80):
        da3, _ = BaseDepthModel.load_depth(da3)
        da2, _ = BaseDepthModel.load_depth(da2)

        da2 = da2.cuda()
        da3 = da3.cuda()

        da2 = normalize(da2)
        da3_math = normalize(convert_to_disparity(da3))
        with torch.inference_mode():
            da3_model_out = normalize(model(da3))

        model_psnr += psnr(da3_model_out, da2)
        math_psnr += psnr(da3_math, da2)

    print("model", model_psnr / len(files), "math", math_psnr / len(files))


if __name__ == "__main__":
    main()
