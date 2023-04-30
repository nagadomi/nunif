# random screentone image generator
# python3 -m waifu2x.training.screentone_generator -n 100 -o ./screentone_test
from PIL import Image, ImageDraw
import random
import numpy as np
import argparse
from tqdm import tqdm
import os
from os import path
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)


def ellipse_rect(center, size):
    return (center[0] - size // 2, center[1] - size // 2,
            center[0] + size // 2, center[1] + size // 2)


def random_crop(x, size):
    i, j, h, w = T.RandomCrop.get_params(x, size)
    x = TF.crop(x, i, j, h, w)
    return x


def random_interpolation(rotate=False):
    interpolations = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
    if rotate:
        interpolations.append(InterpolationMode.NEAREST)
    return random.choice(interpolations)


def gen_mask(size=400):
    dot_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    margin = random.randint(2, dot_size * 2)
    kernel_size = dot_size + margin
    kernel = Image.new("L", (kernel_size, kernel_size), (0,))
    gc = ImageDraw.Draw(kernel)
    gc.ellipse(ellipse_rect((-1, -1), dot_size), fill="white")
    gc.ellipse(ellipse_rect((-1, kernel_size - 1), dot_size), fill="white")
    gc.ellipse(ellipse_rect((kernel_size - 1, -1), dot_size), fill="white")
    gc.ellipse(ellipse_rect((kernel_size - 1, kernel_size - 1), dot_size), fill="white")

    kernel = TF.to_tensor(kernel).squeeze(0)
    p = random.uniform(0, 1)
    if p < 0.4:
        # [o o]
        # [o o]
        repeat_y = repeat_x = (size * 3) // kernel_size
        grid = kernel.squeeze(0).repeat(repeat_y, repeat_x).unsqueeze(0)
        grid = TF.to_pil_image(grid)
        grid = random_crop(grid, (size * 2, size * 2))
    else:
        # [  o  ]
        # [o   o]
        # [  o  ]
        if p < 0.8:
            angle = 45
        else:
            angle = random.uniform(-180, 180)
        repeat_y = repeat_x = (size * 3 * 2) // kernel_size
        grid = kernel.squeeze(0).repeat(repeat_y, repeat_x).unsqueeze(0)
        grid = TF.to_pil_image(grid)
        grid = TF.rotate(grid, angle=angle, interpolation=random_interpolation(rotate=True))
        grid = TF.center_crop(grid, (size * 3, size * 3))
        grid = random_crop(grid, (size * 2, size * 2))

    grid = TF.resize(grid, (size, size), interpolation=random_interpolation(), antialias=True)
    return grid


def gen_color():
    if random.uniform(0, 1) < 0.25:
        # random color
        bg = []
        for _ in range(3):
            bg.append(random.randint(0, 255))
        bg_mean = int(np.mean(bg))
        if bg_mean > 128:
            fg = np.clip([c - random.randint(32, 192) for c in bg], 0, 255)
        else:
            fg = np.clip([c + random.randint(32, 192) for c in bg], 0, 255)

        is_grayscale = random.uniform(0, 1) < 0.5
        if is_grayscale:
            fg_mean = int(np.mean(fg))
            fg = [fg_mean, fg_mean, fg_mean]
            bg = [bg_mean, bg_mean, bg_mean]
    else:
        # black white
        a = random.randint(0, 16)
        b = random.randint(255 - 16, 255)
        bg = [b, b, b]
        fg = [a, a, a]
    return tuple(fg), tuple(bg)


IMAGE_SIZE = 640
WINDOW_SIZE = 400  # 320 < WINDOW_SIZE


def gen():
    size = IMAGE_SIZE
    fg_color, bg_color = gen_color()
    bg = Image.new("RGB", (WINDOW_SIZE, WINDOW_SIZE), bg_color)
    fg = Image.new("RGB", (WINDOW_SIZE, WINDOW_SIZE), fg_color)
    mask = gen_mask(WINDOW_SIZE)
    bg.putalpha(255)
    fg.putalpha(mask)
    window = Image.alpha_composite(bg, fg)

    screen = Image.new("RGB", (size, size), bg_color)
    pad = (screen.height - window.height) // 2
    screen.paste(window, (pad, pad))
    gc = ImageDraw.Draw(screen)
    if random.uniform(0, 1) < 0.5:
        line_width = random.randint(4, 8)
    else:
        line_width = random.randint(3, 12)
    gc.rectangle((pad, pad, pad + window.width, pad + window.height), outline=fg_color, width=line_width)

    return screen


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-samples", "-n", type=int, default=200,
                        help="number of images to generate")
    parser.add_argument("--seed", type=int, default=71, help="random seed")
    parser.add_argument("--postfix", type=str, help="filename postfix")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    postfix = "_" + args.postfix if args.postfix else ""
    for i in tqdm(range(args.num_samples), ncols=80):
        im = gen()
        im.save(path.join(args.output_dir, f"__SCREENTONE_{i}{postfix}.png"))


if __name__ == "__main__":
    main()
