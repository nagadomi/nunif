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


def gen_color(disable_color):
    line_masking = False
    line_overlay = None
    bg2 = None
    if (not disable_color) and random.uniform(0, 1) < 0.25:
        # random color
        bg = []
        for _ in range(3):
            bg.append(random.randint(0, 255))
        bg_mean = int(np.mean(bg))
        if bg_mean > 128:
            fg = np.clip([c - random.randint(32, 192) for c in bg], 0, 255)
        else:
            fg = np.clip([c + random.randint(32, 192) for c in bg], 0, 255)
        line = fg
        is_grayscale = random.uniform(0, 1) < 0.5
        if is_grayscale:
            fg_mean = int(np.mean(fg))
            fg = [fg_mean, fg_mean, fg_mean]
            bg = [bg_mean, bg_mean, bg_mean]
            line = fg
        if random.uniform(0, 1) < 0.1:
            if random.uniform(0, 1) < 0.5:
                line_overlay = tuple(line)
            else:
                line_masking = True
                line_overlay = tuple(bg)
    else:
        # black white
        if random.uniform(0, 1) < 0.8:
            c = random.randint(255 - 16, 255)
            bg = [c, c, c]
        else:
            c = random.randint(255 - 80, 255 - 16)
            bg = [c, c, c]
            c = random.randint(255 - 16, 255)
            bg2 = [c, c, c]
        if disable_color or random.uniform(0, 1) < 0.5:
            # black ink
            c = random.randint(0, 16)
            fg = [c, c, c]
            line = fg
            if random.uniform(0, 1) < 0.1:
                line_masking = True
                line_overlay = tuple(bg)
        else:
            # gray
            if random.uniform(0, 1) < 0.5:
                c = random.randint(0, 180)
            else:
                c = random.randint(80, 180)
            fg = [c, c, c]
            c = random.randint(0, 16)
            line = [c, c, c]
            if random.uniform(0, 1) < 0.1:
                # line overlay
                line_overlay = tuple(line)

    if bg2 is None:
        bg2 = bg
    return tuple(fg), tuple(bg), tuple(bg2), tuple(line), line_overlay, line_masking


def gen_mask(size=400):
    if random.uniform(0, 1) < 0.5:
        dot_size = random.choice([5, 7, 9, 11, 13])
    else:
        dot_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    p = random.uniform(0, 1)
    if p < 0.33:
        margin = random.randint(2, dot_size)
    elif p < 0.66:
        margin = random.randint(2, dot_size * 2)
    else:
        margin = random.choice([7, 9, 11, 13, 15, 17, 19])

    kernel_size = dot_size + margin
    kernel = Image.new("L", (kernel_size, kernel_size), "black")
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
        repeat_y = repeat_x = (size * 2) // kernel_size
        grid = kernel.squeeze(0).repeat(repeat_y, repeat_x).unsqueeze(0)
        grid = TF.to_pil_image(grid)
        grid = random_crop(grid, (size, size))
    else:
        # [  o  ]
        # [o   o]
        # [  o  ]
        if p < 0.8:
            angle = 45
        else:
            angle = random.uniform(-180, 180)
        repeat_y = repeat_x = (size * 4) // kernel_size
        grid = kernel.squeeze(0).repeat(repeat_y, repeat_x).unsqueeze(0)
        grid = TF.to_pil_image(grid)
        grid = TF.rotate(grid, angle=angle, interpolation=random_interpolation(rotate=True))
        grid = TF.center_crop(grid, (size * 2, size * 2))
        grid = random_crop(grid, (size, size))

    return grid


def gen_line_overlay(size, line_scale=1):
    window = Image.new("L", (size * 2, size * 2), "black")
    if random.uniform(0, 1) < 0.5:
        line_width = random.randint(3, 8) * 2
    else:
        line_width = random.randint(3, 16) * 2
    line_width *= line_scale

    if random.uniform(0, 1) < 0.5:
        margin = random.randint(int(line_width * 0.75), line_width * 2)
    else:
        margin = random.randint(2, line_width * 4)
    offset = random.randint(0, 20)

    gc = ImageDraw.Draw(window)
    x = offset
    while x < window.width:
        gc.line(((x, 0), (x, window.height)), fill="white", width=line_width)
        x = x + line_width + margin

    angle = random.uniform(-180, 180)
    window = TF.rotate(window, angle=angle, interpolation=random_interpolation(rotate=True))
    window = TF.center_crop(window, (size, size))

    return window


IMAGE_SIZE = 640
WINDOW_SIZE = 400  # 320 < WINDOW_SIZE


def gen(disable_color):
    fg_color, window_bg_color, bg_color, line_color, line_overlay_color, line_masking = gen_color(disable_color)
    bg = Image.new("RGB", (WINDOW_SIZE * 2, WINDOW_SIZE * 2), window_bg_color)
    fg = Image.new("RGB", (WINDOW_SIZE * 2, WINDOW_SIZE * 2), fg_color)
    mask = gen_mask(WINDOW_SIZE * 2)
    bg.putalpha(255)
    fg.putalpha(mask)
    window = Image.alpha_composite(bg, fg)
    if line_overlay_color is not None:
        mask = gen_line_overlay(WINDOW_SIZE * 2, line_scale=4 if line_masking else 1)
        fg = Image.new("RGB", (WINDOW_SIZE * 2, WINDOW_SIZE * 2), line_overlay_color)
        window.putalpha(255)
        fg.putalpha(mask)
        window = Image.alpha_composite(window, fg)

    screen = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE * 2), bg_color)
    pad = (screen.height - window.height) // 2
    screen.paste(window, (pad, pad))
    gc = ImageDraw.Draw(screen)
    if random.uniform(0, 1) < 0.5:
        line_width = random.randint(4, 8) * 2
    else:
        line_width = random.randint(3, 12) * 2
    gc.rectangle((pad, pad, pad + window.width, pad + window.height), outline=line_color, width=line_width)

    screen = TF.resize(screen, (IMAGE_SIZE, IMAGE_SIZE), interpolation=random_interpolation(), antialias=True)

    return screen


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-samples", "-n", type=int, default=200,
                        help="number of images to generate")
    parser.add_argument("--seed", type=int, default=71, help="random seed")
    parser.add_argument("--postfix", type=str, help="filename postfix")
    parser.add_argument("--use-color", action="store_true", help="use random RGB color")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    postfix = "_" + args.postfix if args.postfix else ""
    for i in tqdm(range(args.num_samples), ncols=80):
        im = gen(disable_color=not args.use_color)
        im.save(path.join(args.output_dir, f"__SCREENTONE_{i}{postfix}.png"))


if __name__ == "__main__":
    main()
