from PIL import Image
import argparse
import os
from os import path


def show(qtables):
    for idx, dct_coef in qtables.items():
        print(f"{idx}:")
        for i in range(8):
            row = [dct_coef[i * 8 + j] for j in range(8)]
            print(row)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, help="input jpeg file")
    args = parser.parse_args()
    if args.input is not None:
        with Image.open(args.input) as im:
            show(im.quantization)
    else:
        image_dir = path.join(path.dirname(__file__), "images")
        for fn in sorted(os.listdir(image_dir)):
            if fn.endswith(".jpg"):
                with Image.open(path.join(image_dir, fn)) as im:
                    print(f"* {fn}")
                    show(im.quantization)


if __name__ == "__main__":
    main()
