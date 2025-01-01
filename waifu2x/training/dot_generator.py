# random dot image generator
# port from waifu2x/image_generator/dot

from PIL import Image, ImageDraw, ImageOps
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
DOT_SCALE_PREFIX = {2: "_DOT_2x_", 4: "_DOT_4x_"}


def exec_prob(prob):
    return random.uniform(0, 1) < prob


def highcolor(black):
    v = random.uniform(0, 0.08)
    if black:
        return v
    else:
        return 1 - v


def gen_color(black):
    if exec_prob(0.2):
        return (highcolor(black), highcolor(black), highcolor(black))
    else:
        rgb = []
        for _ in range(3):
            if exec_prob(0.3):
                v = highcolor(black)
            else:
                v = random.uniform(0, 1)
            rgb.append(v)
        return tuple(rgb)


def remove_alpha(im, bg_color):
    nobg = Image.new(im.mode[:-1], im.size, bg_color)
    nobg.paste(im, im.getchannel("A"))
    return nobg


def gen_dot_block(block_size=24, scale=1, rotate=False, bg_color=None, bg_color_black=True):
    block = np.zeros((block_size, block_size, 3), dtype=np.float32)
    y_margin = random.randint(1, 3)
    if rotate:
        size = random.randint(3, 5)
        use_cross_and_skip = False
    else:
        size = random.randint(1, 5)
        use_cross_and_skip = exec_prob(0.25)
    use_random_size = exec_prob(0.75)

    xm_shift = random.randint(0, 1)
    xm = random.randint(1, 4)
    ym = random.randint(1, 4)

    def mod(y, x):
        return (x + (y % 2) * xm_shift) % xm == 0 and y % ym == 0

    if bg_color is not None:
        bg = bg_color
        fg = gen_color(black=bg_color_black)
    else:
        if exec_prob(0.5):
            fg = gen_color(black=False)
            bg = gen_color(black=True)
        else:
            fg = gen_color(black=True)
            bg = gen_color(black=False)

    block[:, :] = bg
    for y in range(y_margin, block_size - y_margin):
        if use_random_size:
            if rotate:
                size = random.randint(3, 5)
            else:
                size = random.randint(1, 5)
        yc = math.floor(y / size)
        b = 0
        if use_cross_and_skip and exec_prob(0.5):
            b = random.randint(0, 1)
        for x in range(y_margin, block_size - y_margin):
            xc = math.floor(x / size)
            if use_cross_and_skip:
                if exec_prob(0.75) and mod(yc + b, xc + b):
                    block[y, x, :] = fg
            else:
                if mod(yc + b, xc + b):
                    block[y, x, :] = fg

    block = (block * 255).astype(np.uint8)
    im = Image.fromarray(block)
    if exec_prob(0.5):
        # flip
        if exec_prob(0.5):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.transpose(random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]))

    im = im.resize((block_size * scale, block_size * scale), resample=Image.Resampling.NEAREST)
    if rotate:
        im = im.convert("RGBA")
        im = im.rotate(random.randint(0, 90), resample=Image.Resampling.BILINEAR)
        im = remove_alpha(im, (int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255)))
    return im


def draw_line(block, p1, p2, fg, size, size_step):
    steps = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if not size_step:
        pass
    else:
        steps = steps / size
    steps = int(steps)
    if steps == 0:
        return p1

    y_delta = (p2[0] - p1[0]) / steps
    x_delta = (p2[1] - p1[1]) / steps
    p = p1
    last_p = None
    for _ in range(steps):
        last_p = [int(p[0]), int(p[1])]
        for y in range(size):
            y = min(y + last_p[0], block.shape[0] - 1)
            for x in range(size):
                x = min(x + last_p[1], block.shape[1] - 1)
                block[y, x, :] = fg
        p[0] += y_delta
        p[1] += x_delta
    return last_p


def draw_random_line(block, fg, size, size_step):
    block_size = block.shape[0]
    num_points = random.randint(2, 12)
    points = []
    for i in range(num_points):
        points.append([random.randint(0, block_size - 1), random.randint(0, block_size - 1)])
    if exec_prob(0.5):
        if exec_prob(0.5):
            points = sorted(points, key=lambda p: p[0])
        else:
            points = sorted(points, key=lambda p: p[1])
    else:
        random.shuffle(points)

    p = points[0]
    for next_p in points[1:]:
        p = draw_line(block, p, next_p, fg, size, size_step=size_step)


def gen_dot_line_block(block_size=24, scale=1, rotate=False, bg_color=None, bg_color_black=True):
    block = np.zeros((block_size, block_size, 3), dtype=np.float32)
    margin = random.randint(1, 3)
    if rotate:
        size = random.randint(3, 5)
    else:
        if exec_prob(0.5):
            size = random.randint(1, 5)
        else:
            size = random.randint(1, 3)

    if bg_color is not None:
        bg = bg_color
        fg1 = gen_color(black=bg_color_black)
        fg2 = gen_color(black=bg_color_black)
    else:
        if exec_prob(0.5):
            fg1 = gen_color(black=False)
            fg2 = gen_color(black=False)
            bg = gen_color(black=True)
        else:
            fg1 = gen_color(black=True)
            fg2 = gen_color(black=True)
            bg = gen_color(black=False)

    block[:, :] = bg
    if exec_prob(0.5):
        p = draw_random_line(block, fg1, size, size_step=(rotate or exec_prob(0.5)))
    else:
        p = draw_random_line(block, fg1, size, size_step=(rotate or exec_prob(0.5)))
        p = draw_random_line(block, fg2, size, size_step=(rotate or exec_prob(0.5)))

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
    bg_color_black = exec_prob(0.5)
    bg_color = gen_color(bg_color_black)
    blocks = []
    for _ in range(cols * cols):
        if exec_prob(0.2):
            block = gen_dot_line_block(block_size=block_size, scale=scale, rotate=rotate,
                                       bg_color=bg_color, bg_color_black=bg_color_black)
        else:
            block = gen_dot_block(block_size=block_size, scale=scale, rotate=rotate,
                                  bg_color=bg_color, bg_color_black=bg_color_black)
        blocks.append(block)
    im = image_grid(blocks, block_size * scale, cols, cols)
    return im


COLS_MAP = {2: 4, 4: 2, 8: 1}


def gen(cols_scale=1, rotate=False, dot_scale=2):
    assert isinstance(cols_scale, int)
    assert dot_scale in {2, 4}
    line_block = exec_prob(0.2)
    if dot_scale == 2:
        scale = random.choices((2, 4, 8), weights=(0.25, 1, 1), k=1)[0]
    elif dot_scale == 4:
        scale = random.choices((4, 8), weights=(1, 1), k=1)[0]
    cols = COLS_MAP[scale]
    block_size = random.choice([40, 40, 40, 20, 20, 10])
    block_size_scale = 40 // block_size
    return gen_dot_grid(block_size, scale, cols * cols_scale * block_size_scale, rotate=rotate)


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


def ellipse_rect(center, size):
    return (center[0] - size // 2, center[1] - size // 2,
            center[0] + size // 2, center[1] + size // 2)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--size", "-s", type=int, default=640, choices=[320, 640, 1280],
                        help="image size")
    parser.add_argument("--dot-scale", type=int, default=2, choices=[2, 4],
                        help="minimum dot size")
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
        rotate = random.choice([True, False, False]) if args.rotate else False
        dot = gen(cols_scale=cols_scale, rotate=rotate, dot_scale=args.dot_scale)

        hole = random.choice([True, False, False])
        rotate2 = random.choice([True, False, False, False]) if args.rotate else False
        rotate = rotate or rotate2
        if hole:
            hole_scale = 4 if rotate else 1
            mask = Image.new("L", (dot.width * hole_scale, dot.height * hole_scale), (0,))
            gc = ImageDraw.Draw(mask)
            color = tuple([random.randint(0, 255) for _ in range(3)] + [255])
            circle_hole = random.choice([True, False])
            if circle_hole:
                r = random.randint(int(mask.width * 0.25), int(mask.width * 0.75))
                center = [random.randint(int(-mask.width * 0.5), int(mask.width + mask.width * 0.5)),
                          random.randint(int(-mask.height * 0.5), int(mask.height + mask.height * 0.5))]
                gc.ellipse(ellipse_rect(center, r), fill=255)
            else:
                p1 = (random.randint(0, int(mask.width * 0.5)),
                      random.randint(0, int(mask.height * 0.5)))
                p2 = (random.randint(int(mask.width * 0.5), mask.width),
                      random.randint(0, int(mask.height * 0.5)))
                p3 = (random.randint(int(mask.width * 0.5), mask.width),
                      random.randint(int(mask.height * 0.5), mask.height))
                p4 = (random.randint(0, int(mask.width * 0.5)),
                      random.randint(int(mask.height * 0.5), mask.height))
                gc.polygon((p1, p2, p3, p4), fill=255)
            if hole_scale != 1:
                mask = mask.resize(dot.size, resample=Image.Resampling.LANCZOS)
            if random.choice([True, True, True, False]):
                mask = ImageOps.invert(mask)
            bg = Image.new("RGB", dot.size, color)
            dot = Image.composite(dot, bg, mask)

        if rotate2:
            dot = dot.rotate(random.randint(-180, 180), resample=Image.Resampling.BILINEAR)
        dot_prefix = DOT_SCALE_PREFIX[args.dot_scale]
        if rotate:
            output_filename = path.join(args.output_dir, f"_{dot_prefix}ROTATE_{i}{postfix}.png")
        else:
            output_filename = path.join(args.output_dir, f"{NEAREST_PREFIX}{dot_prefix}{i}{postfix}.png")

        dot.save(output_filename)


if __name__ == "__main__":
    main()
    # _show()
