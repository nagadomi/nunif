# Measure the ratio of white pixels and black pixels in a mask image
# black/white is about 900
import argparse
from nunif.utils.image_loader import ImageLoader
from torchvision.io import read_image
import random


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input dir")

    args = parser.parse_args()

    files = [fn for fn in ImageLoader.listdir(args.input) if fn.endswith("_ML.png")]
    random.shuffle(files)
    fg_sum = 0
    bg_sum = 0
    i = 0
    for fn in files:
        x = read_image(fn).float().sum(dim=0)
        fg_count = (x > 0).sum().item()
        bg_count = (x.numel() - fg_count)

        fg_sum += fg_count
        bg_sum += bg_count
        i += 1
        if i % 1000 == 0:
            print("balck/white = ", bg_sum / fg_sum)

    print("balck/white = ", bg_sum / fg_sum)


if __name__ == "__main__":
    main()
