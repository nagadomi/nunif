# random dot image generator
# port from waifu2x/image_generator/dot

from PIL import Image
import random
import numpy as np
import math
import argparse
from tqdm import tqdm
import os
from os import path


# File name prefix for specifying the downsampling filter
# This used in dataset.py
NEAREST_PREFIX = "__NEAREST_"


def exec_prob(prob):
    return random.uniform(0, 1) < prob


def gen_color(black):
    if exec_prob(0.2):
        if black:
            return (0.0, 0.0, 0.0)
        else:
            return (1.0, 1.0, 1.0)
    else:
        rgb = []
        for _ in range(3):
            if exec_prob(0.3):
                v = float(random.randint(0, 1))
            else:
                v = random.uniform(0, 1)
            rgb.append(v)
        return tuple(rgb)


def remove_alpha(im, bg_color):
    nobg = Image.new(im.mode[:-1], im.size, bg_color)
    nobg.paste(im, im.getchannel("A"))
    return nobg


def gen_dot_block(block_size=24, scale=1, rotate=False):
    block = np.zeros((block_size, block_size, 3), dtype=np.float32)
    margin = random.randint(1, 3)
    if rotate:
        size = random.randint(3, 5)
        use_cross_and_skip = False
    else:
        size = random.randint(1, 5)
        use_cross_and_skip = exec_prob(0.5)

    xm = random.randint(2, 4)
    ym = random.randint(2, 4)

    def mod(x, y):
        return x % xm == 0 and y % ym == 0

    if exec_prob(0.5):
        fg = gen_color(black=False)
        bg = gen_color(black=True)
    else:
        fg = gen_color(black=True)
        bg = gen_color(black=False)

    block[:, :] = bg
    for y in range(margin, block_size - margin):
        yc = math.floor(y / size)
        b = 0
        if use_cross_and_skip and exec_prob(0.5):
            b = random.randint(0, 1)
        for x in range(margin, block_size - margin):
            xc = math.floor(x / size)
            if use_cross_and_skip:
                if exec_prob(0.75) and mod(yc + b, xc + b):
                    block[y, x, :] = fg
            else:
                if mod(yc + b, xc + b):
                    block[y, x, :] = fg

    block = (block * 255).astype(np.uint8)
    im = Image.fromarray(block)
    im = im.resize((block_size * scale, block_size * scale), resample=Image.Resampling.NEAREST)
    if rotate:
        im = im.convert("RGBA")
        im = im.rotate(random.randint(0, 90), resample=Image.Resampling.BILINEAR)
        im = remove_alpha(im, (int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255)))
    return im


def image_grid(blocks, block_size, rows, cols):
    assert len(blocks) == rows * cols
    w, h = block_size, block_size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, block in enumerate(blocks):
        grid.paste(block, (i % cols * w, i // cols * h))
    return grid


def gen_dot_grid(block_size, scale, cols, rotate=False):
    blocks = []
    for _ in range(cols * cols):
        block = gen_dot_block(block_size=block_size, scale=scale, rotate=rotate)
        blocks.append(block)
    im = image_grid(blocks, block_size * scale, cols, cols)
    return im


COLS_MAP = {2: 4, 4: 2, 8: 1}


def gen(cell=40, cols_scale=1, rotate=False):
    assert isinstance(cols_scale, int)
    scale = random.choices((2, 4, 8), weights=(0.25, 1, 1), k=1)[0]
    cols = COLS_MAP[scale]
    return gen_dot_grid(cell, scale, cols * cols_scale, rotate=rotate)


def _validate():
    for _ in range(100):
        dot = gen()
        dot_half = dot.resize((dot.size[0] // 2, dot.size[1] // 2),
                              resample=Image.Resampling.NEAREST)
        dot2x = dot_half.resize((dot_half.size[0] * 2, dot_half.size[1] * 2),
                                resample=Image.Resampling.NEAREST)
        diff = np.array(dot, dtype=np.float32) - np.array(dot2x, dtype=np.float32)
        diff_sum = (diff * diff).sum()
        assert diff_sum < 0.0001
    print("OK")


def _show():
    gen_dot_grid(40, 2, 8).show()
    gen_dot_grid(40, 4, 4).show()
    gen_dot_grid(40, 8, 2).show()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--size", "-s", type=int, default=640, choices=[320, 640, 1280],
                        help="image size")
    parser.add_argument("--num-samples", "-n", type=int, default=200,
                        help="number of images to generate")
    parser.add_argument("--rotate", action="store_true",
                        help="allow rotation")
    parser.add_argument("--seed", type=int, default=71,
                        help="random seed")
    parser.add_argument("--postfix", type=str, help="filename postfix")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()
    random.seed(args.seed)
    cols_scale = {320: 1, 640: 2, 1280: 4}[args.size]
    postfix = "_" + args.postfix if args.postfix else ""
    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(args.num_samples), ncols=80):
        rotate = random.choice([True, False]) if args.rotate else False
        dot = gen(cols_scale=cols_scale, rotate=rotate)
        if rotate:
            output_filename = path.join(args.output_dir, f"__DOT_ROTATE_{i}{postfix}.png")
        else:
            output_filename = path.join(args.output_dir, f"{NEAREST_PREFIX}_DOT_{i}{postfix}.png")

        dot.save(output_filename)


if __name__ == "__main__":
    main()
    # _show()
