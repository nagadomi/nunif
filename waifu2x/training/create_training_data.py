import os
import sys
import argparse
from os import path
from tqdm import tqdm
import random
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import list_images
from nunif.transforms.std import pad
from multiprocessing import cpu_count


def split_image(filepath_prefix, im, size, stride, reject_rate, format):
    w, h = im.size
    rects = []
    for y in range(0, h, stride):
        if not y + size <= h:
            break
        for x in range(0, w, stride):
            if not x + size <= w:
                break
            rect = TF.crop(im, y, x, size, size)
            center = TF.center_crop(rect, (size // 2, size // 2))
            color_stdv = TF.to_tensor(center).std(dim=[1, 2]).sum().item()
            rects.append((rect, color_stdv))

    n_reject = int(len(rects) * reject_rate)
    rects = [v[0] for v in sorted(rects, key=lambda v: v[1], reverse=True)][0:len(rects) - n_reject]

    index = 0
    for rect in rects:
        if format == "png":
            rect.save(f"{filepath_prefix}_{index}.png")
        elif format == "webp":
            rect.save(f"{filepath_prefix}_{index}.webp", lossless=True)
        else:
            raise ValueError(f"format {format}")
        index += 1


class CreateTrainingData(Dataset):
    def __init__(self, input_dir, output_dir, args):
        super().__init__()
        self.files = list_images(input_dir)
        self.args = args
        self.filename_prefix = args.prefix + "_" if args.prefix else ""
        self.output_dir = output_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        bg_color = random.randint(0, 255)
        im, _ = load_image_simple(filename, color="rgb", bg_color=bg_color)
        if im is None:
            return -1
        if self.args.pad:
            bg = random.randint(0, 255)
            im = pad(im, [self.args.size] * 2, mode=self.args.pad_mode, fill=bg)
        split_image(
            path.join(self.output_dir, self.filename_prefix + str(i)),
            im, self.args.size, int(self.args.size * self.args.stride), self.args.reject_rate,
            self.args.format,
        )
        im.close()

        return 0


def main(args):
    num_workers = cpu_count()

    for dataset_type in ("eval", "train"):
        input_dir = path.join(args.dataset_dir, dataset_type)
        output_dir = path.join(args.data_dir, dataset_type)
        if not path.exists(input_dir):
            print(f"Error: `{input_dir}` not found", file=sys.stderr)
            return

    for dataset_type in ("eval", "train"):
        print(f"** {dataset_type}")
        input_dir = path.join(args.dataset_dir, dataset_type)
        output_dir = path.join(args.data_dir, dataset_type)

        os.makedirs(output_dir, exist_ok=True)
        loader = DataLoader(
            CreateTrainingData(input_dir, output_dir, args),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=8,
            drop_last=False
        )
        for _ in tqdm(loader, ncols=80):
            pass


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--size", type=int, default=640,
                        help="image size")
    parser.add_argument("--stride", type=float, default=0.25,
                        help="stride_size = int(size * stride)")
    parser.add_argument("--reject-rate", type=float, default=0.5,
                        help="reject rate for hard example mining")
    parser.add_argument("--prefix", type=str, default="",
                        help="prefix for output filename")
    parser.add_argument("--format", type=str, choices=["png", "webp"], default="png",
                        help="output image format")
    parser.add_argument("--pad", action="store_true",
                        help="use padding for small images")
    parser.add_argument("--pad-mode", choices=["reflect", "edge", "constant"], default="reflect",
                        help="padding mode for pad")
    parser.set_defaults(handler=main)

    return parser
