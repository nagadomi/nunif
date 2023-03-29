from PIL import Image
import os
from os import path
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import io
import json
import argparse
import time


TEST_IMAGE = path.join(path.dirname(__file__), "images", "donut_noisy.png")
OUTPUT_DIR = path.relpath(path.join(path.dirname(__file__), "..", "..",
                                    "tmp", "search_qtable"))


CHECKBOARD_KERNEL = torch.tensor([
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1],
], dtype=torch.float32).reshape(1, 1, 8, 8)

# v/h could be reversed
VERTICAL_LINE_KERNEL = torch.tensor([
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
    [0, 0, 0.5, -1, 0.5, 0, 0, 0],
], dtype=torch.float32).reshape(1, 1, 8, 8)
HORIZONTAL_LINE_KERNEL = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=torch.float32).reshape(1, 1, 8, 8)


ZIGZAG_SCAN_INDEX = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4,
    5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7,
    14, 21, 28, 35, 42, 49, 56,
    57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53,
    60, 61, 54, 47, 55, 62, 63]


FIXED_UV_QTABLE = [
    7, 7, 10, 19, 32, 32, 32, 32,
    7, 8, 10, 26, 32, 32, 32, 32,
    10, 10, 22, 32, 32, 32, 32, 32,
    19, 26, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
]


def calc_score(im, qtable, uv_qtable, kernels, criterion):
    with io.BytesIO() as buff:
        im.save(buff, format="jpeg",
                qtables={0: qtable, 1: uv_qtable}, subsampling="4:2:0")
        buff.seek(0)
        im = Image.open(buff)
        x = TF.to_tensor(TF.to_grayscale(im)).unsqueeze(0)
        score = 0
        for kernel in kernels:
            if criterion == "max":
                 score += F.adaptive_max_pool2d(
                     F.conv2d(x, weight=kernel, stride=1, padding=0),
                     (16, 16)
                 ).mean().item()
            else:
                # score += F.conv2d(x, weight=kernel, stride=1, padding=0).abs().mean().item()
                score += F.conv2d(x, weight=kernel, stride=1, padding=0).mean().item()
        return score


def save_best(output_dir, name, im, qtable, uv_qtable):
    im.save(path.join(output_dir, f"{name}.jpg"), format="jpeg",
            qtables={0: qtable, 1: uv_qtable}, subsampling="4:2:0")
    with open(path.join(output_dir, f"{name}.json"), mode="w") as f:
        f.write(json.dumps({0: qtable, 1: uv_qtable}))


def change_qtable(qtable, n, protect_index):
    qtable = qtable[:]
    for _ in range(n):
        i = random.randint(protect_index + 1, 64 - 1)
        qtable[ZIGZAG_SCAN_INDEX[i]] = random.randint(0, 255)
    return qtable


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, default=TEST_IMAGE,
                        help="test image file")
    parser.add_argument("--output-dir", "-o", type=str, default=OUTPUT_DIR,
                        help="output directory")
    parser.add_argument("--target", type=str, default="checkboard",
                        choices=["vline", "hline", "vhline", "checkboard"],
                        help="target patterns")
    parser.add_argument("--protect-index", type=int, default=10,
                        help="protect qtable index to reduce block noise. 6, 10, 15 are reasonable values.")
    parser.add_argument("--protect-value", type=int, default=10,
                        help="qtable value for protect indexes. 1-255: 1=quality 100, 255=quality 0")
    parser.add_argument("--max-epoch", type=int, default=80000,
                        help="max epoch")
    parser.add_argument("--criterion", type=str, choices=["mean", "max"], default="max",
                        help="criterion")
    parser.add_argument("--fixed-uv", action="store_true", 
                        help="use fixed qtable for uv")
    parser.add_argument("--fixed-uv-value", type=int, default=40,
                        help="qtable value for fixed uv qtable")

    args = parser.parse_args()
    runtime_name = f"{args.target}_{int(time.time())}"
    if args.target == "vline":
        kernels = [VERTICAL_LINE_KERNEL]
    elif args.target == "hline":
        kernels = [HORIZONTAL_LINE_KERNEL]
    elif args.target == "vhline":
        kernels = [HORIZONTAL_LINE_KERNEL, VERTICAL_LINE_KERNEL]
    elif args.target == "checkboard":
        kernels = [CHECKBOARD_KERNEL]
    else:
        raise NotImplementedError()
    num_changes = [1, 2, 3, 4, 8, 16]
    for i in range(15, 64 - 1):
        FIXED_UV_QTABLE[ZIGZAG_SCAN_INDEX[i]] = args.fixed_uv_value

    # init
    test_image = Image.open(args.input)
    test_image = test_image.convert("RGB")
    os.makedirs(args.output_dir, exist_ok=True)
    qtable = [args.protect_value for _ in range(64)]
    uv_qtable = FIXED_UV_QTABLE if args.fixed_uv else qtable
    best_score = calc_score(test_image, qtable, uv_qtable, kernels, args.criterion)

    # search best(worst) qtable
    print(f"Start output_dir={args.output_dir}, name={runtime_name}")
    for epoch in range(args.max_epoch):
        n = random.choice(num_changes)
        new_qtable = change_qtable(qtable, n=n, protect_index=args.protect_index)
        uv_qtable = FIXED_UV_QTABLE if args.fixed_uv else new_qtable
        new_score = calc_score(test_image, new_qtable, uv_qtable, kernels, args.criterion)
        if new_score > best_score:
            qtable[:] = new_qtable[:]
            best_score = new_score
            save_best(args.output_dir, runtime_name, test_image, new_qtable, uv_qtable)
            print(f"update score epoch={epoch}: {best_score}")
    print("done")


if __name__ == "__main__":
    main()
