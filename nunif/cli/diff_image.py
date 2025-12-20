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
    parser.add_argument("--resize", action="store_true", help="allow resizing to fit the image")
    args = parser.parse_args()
    if len(args.input) != 2:
        raise ValueError("Specify two files")

    if path.isfile(args.input[0]) and path.isfile(args.input[1]):
        im1, _ = load_image_simple(args.input[0], color="any")
        im2, _ = load_image_simple(args.input[1], color="any")
        if not compare_size(im1, im2):
            if not args.resize:
                print("size differ")
            else:
                im1 = TF.resize(im1, size=(im2.height, im2.width))
                print(f"PSNR: {calc_psnr(im1, im2)}")
        else:
            print(f"PSNR: {calc_psnr(im1, im2)}")
    elif path.isfile(args.input[0]) and path.isdir(args.input[1]):
        im1, _ = load_image_simple(args.input[0], color="any")
        files2 = ImageLoader.listdir(args.input[1])

        psnr_max = 0
        psnr_max_file = None
        psnr_min = 1000
        psnr_min_file = None

        for i in range(len(files2)):
            im2, _ = load_image_simple(files2[i], color="any")
            if not compare_size(im1, im2):
                if not args.resize:
                    psnr = float("nan")
                else:
                    im1 = TF.resize(im1, size=(im2.height, im2.width))
                    psnr = calc_psnr(im1, im2)
            else:
                psnr = calc_psnr(im1, im2)
            if math.isnan(psnr):
                print(f"size differ {path.basename(files2[i])}")
            else:
                print(f"PSNR: {psnr} {path.basename(files2[i])}")
                if psnr_max < psnr:
                    psnr_max = psnr
                    psnr_max_file = files2[i]
                if psnr_min > psnr:
                    psnr_min = psnr
                    psnr_min_file = files2[i]
        print(f"Max PSNR: {psnr_max}: {psnr_max_file}")
        print(f"Min PSNR: {psnr_min}: {psnr_min_file}")

    elif path.isdir(args.input[0]) and path.isdir(args.input[1]):
        files1 = ImageLoader.listdir(args.input[0])
        files2 = ImageLoader.listdir(args.input[1])
        assert len(files1) == len(files2)
        # sorted
        psnr_sum = 0
        for i in range(len(files1)):
            im1, _ = load_image_simple(files1[i], color="any")
            im2, _ = load_image_simple(files2[i], color="any")

            if not compare_size(im1, im2):
                if not args.resize:
                    psnr = float("nan")
                else:
                    im1 = TF.resize(im1, size=(im2.height, im2.width))
                    psnr = calc_psnr(im1, im2)
            else:
                psnr = calc_psnr(im1, im2)
            psnr_sum += psnr
            if args.verbose:
                if math.isnan(psnr):
                    print(f"size differ {path.basename(files1[i])}")
                else:
                    print(f"PSNR: {psnr} {path.basename(files1[i])}")

        print(f"Mean PSNR: {round(psnr_sum / len(files1), 3)}")
    else:
        raise ValueError("input = `file file` or `file dir` or `dir dir`")


if __name__ == "__main__":
    main()
