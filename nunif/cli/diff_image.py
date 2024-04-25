# A simple image diff tool
#
# python -m nunif.cli.diff_image -i image1.png image2.png
# python -m nunif.cli.diff_image -v -i dir1/ dir2/
import argparse
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple, to_tensor
import torchvision.transforms.functional as TF
import torch
from os import path
import math


def calc_psnr(im1, im2):
    im1 = to_tensor(im1)
    im2 = to_tensor(im2)

    mse = ((im1 - im2) ** 2).mean()
    return round(float(10 * torch.log10(1.0 / (mse + 1.0e-6))), 3)


def compare_size(im1, im2):
    return im1.mode == im2.mode and im1.width == im2.width and im1.height == im2.height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="input file or directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="output result for each images")
    args = parser.parse_args()
    if len(args.input) != 2:
        raise ValueError("Specify two files")

    if path.isfile(args.input[0]):
        im1, _ = load_image_simple(args.input[0], color="any")
        im2, _ = load_image_simple(args.input[1], color="any")
        if not compare_size(im1, im2):
            print("size differ")
        else:
            print(f"PSNR: {calc_psnr(im1, im2)}")
    else:
        files1 = ImageLoader.listdir(args.input[0])
        files2 = ImageLoader.listdir(args.input[1])
        assert len(files1) == len(files2)
        # sorted
        psnr_sum = 0
        for i in range(len(files1)):
            im1, _ = load_image_simple(files1[i], color="any")
            im2, _ = load_image_simple(files2[i], color="any")
            if not compare_size(im1, im2):
                psnr = float("nan")
            else:
                psnr = calc_psnr(im1, im2)
            psnr_sum += psnr
            if args.verbose:
                if math.isnan(psnr):
                    print(f"size differ {path.basename(files1[i])}")
                else:
                    print(f"PSNR: {psnr} {path.basename(files1[i])}")

        print(f"Mean PSNR: {round(psnr_sum/len(files1), 3)}")


if __name__ == "__main__":
    main()
