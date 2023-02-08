# random text image generator
# # DEBUG=1 python3 -m waifu2x.training.text_image_generator -n 100 -o ./text_test --bg-dir /bg/eval --seed 73
from PIL import Image, ImageDraw
import random
import math
import argparse
from tqdm import tqdm
import os
from os import path
from multiprocessing import cpu_count
import threading
import numpy as np
from types import SimpleNamespace
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as TT
import nunif.transforms as NT
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import list_images
from nunif.logger import logger
from nunif.initializer import set_seed
from text_resource.aozora.db import AozoraDB
from text_resource.aozora import utils as AU
from font_resource.metadata import DEFAULT_FONT_NAMES, DEFAULT_FONT_DIR
from font_resource.utils import load_fonts
from font_resource.draw import SimpleLineDraw


def exec_prob(prob):
    return random.uniform(0, 1) < prob


class BackgroundImageGenerator():
    def __init__(self, size, bg_dir):
        assert isinstance(size, int)
        self.size = size
        self.bg_dir = bg_dir
        self.images = list_images(bg_dir)
        assert len(self.images) > 0
        logger.debug(f"BackgroundImageGenerator: {len(self.images)} images")

        self.transforms = TT.Compose([
            NT.ReflectionResize(size),
            TT.RandomApply([TT.ColorJitter(brightness=0.3, hue=0.1)], p=0.25),
            TT.RandomInvert(p=0.125),
            TT.RandomAutocontrast(p=0.25),
            TT.RandomGrayscale(p=0.25),
        ])

    def generate(self):
        bg, _ = load_image_simple(random.choice(self.images), color="rgb")
        return self.transforms(bg)


class TextGenerator():
    AOZORA_AUTHORS = ("夢野 久作", "太宰 治", "芥川 竜之介", "夏目 漱石", "森 鴎外")

    def __init__(self):
        db = AozoraDB()
        items = []
        for author in TextGenerator.AOZORA_AUTHORS:
            ret = db.find_by_author(author, modern_only=True, size_order=True, limit=20)
            items += ret
            logger.debug(f"TextGenerator: {author}: {len(ret)} items")
        lines = []
        for item in items:
            lines += AU.load_speech_lines(item.file_path, remove_punct=True, min_len=4)
        self.lines = lines
        logger.debug(f"TextGenerator: {len(self.lines)} lines")

    def generate(self, drawable):
        retry_count = 0
        while True:
            line = random.choice(self.lines)
            if drawable(line):
                break
            retry_count += 1
            if retry_count > 100:
                raise RuntimeError("Unable to generate drawable text")
        return line


class TextImageGenerator(Dataset):
    def __init__(self, args):
        super().__init__()
        tmp_size = int(args.size * 1.25)
        self.safe_size = int(tmp_size * math.sqrt(2)) + 4
        self.fonts = load_fonts(args.font_names, font_dir=args.font_dir)
        self.text_gen = TextGenerator()
        if args.bg_dir is not None:
            self.bg_gen = BackgroundImageGenerator(self.safe_size, args.bg_dir)
        else:
            self.bg_gen = None
        self.num_samples = args.num_samples
        self.lock = threading.RLock()
        self.transforms = TT.Compose([
            TT.RandomChoice([
                NT.Identity(),
                TT.RandomRotation((-45, 45), interpolation=TT.InterpolationMode.BILINEAR, expand=True),
                TT.RandomPerspective(distortion_scale=1 - 1 / math.sqrt(2), p=1.0),
            ], p=[5, 1, 1]),
            TT.CenterCrop(args.size),
            # TT.CenterCrop(tmp_size),
            # TT.Resize(args.size, interpolation=TT.InterpolationMode.BILINEAR)
        ])

    def gen_basecolor(self):
        if random.uniform(0, 1) < 0.7:
            # random color
            bg = []
            for _ in range(3):
                bg.append(random.randint(0, 255))
            bg_mean = int(np.mean(bg))
            if bg_mean > 128:
                fg = np.clip([c - random.randint(32, 192) for c in bg], 0, 255)
            else:
                fg = np.clip([c + random.randint(32, 192) for c in bg], 0, 255)

            is_grayscale = exec_prob(0.5)
            if is_grayscale:
                fg_mean = int(np.mean(fg))
                fg = [fg_mean, fg_mean, fg_mean]
                bg = [bg_mean, bg_mean, bg_mean]
        else:
            # black white
            a = random.randint(0, 10)
            b = random.randint(245, 255)
            if random.uniform(0, 1) < 0.5:
                bg = [a, a, a]
                fg = [b, b, b]
            else:
                bg = [b, b, b]
                fg = [a, a, a]

        return tuple(fg), tuple(bg)

    def gen_bg(self, bg_color):
        shadow_color = None
        if self.bg_gen is not None and exec_prob(0.5):
            bg = self.bg_gen.generate()
            bg_alpha = random.uniform(0.2, 1)
            bg_color_im = Image.new("RGB", bg.size, bg_color)
            bg = Image.blend(bg, bg_color_im, bg_alpha)
            if bg_alpha < 0.5:
                shadow_color = bg_color
            elif exec_prob(0.5):
                shadow_color = bg_color
            return bg, shadow_color
        else:
            bg = Image.new("RGB", (self.safe_size, self.safe_size), bg_color)
            return bg, shadow_color

    def gen_config(self):
        fg_color, bg_color = self.gen_basecolor()
        font = random.choice(self.fonts)
        if exec_prob(0.85):
            font_size = random.randint(16, 64)
        else:
            font_size = random.randint(64, 256)
        letter_spacing = int(random.uniform(0, 0.2) * font_size)
        line_spacing = int(random.uniform(0.2, 0.5) * font_size)
        vertical = exec_prob(0.5)
        bg, shadow_color = self.gen_bg(bg_color)
        shadow_width = 2 + random.randint(0, font_size // 8)

        return SimpleNamespace(fg_color=fg_color, shadow_color=shadow_color, shadow_width=shadow_width,
                               bg=bg,
                               font=font, font_size=font_size, vertical=vertical,
                               letter_spacing=letter_spacing, line_spacing=line_spacing)

    def gen_text_block_image(self):
        conf = self.gen_config()
        canvas = conf.bg
        gc = ImageDraw.Draw(canvas)
        pen = SimpleLineDraw(conf.font, font_size=conf.font_size, vertical=conf.vertical)
        margin = conf.font_size // 2
        x = y = margin
        if conf.vertical:
            while x + conf.font_size + conf.line_spacing + margin < canvas.size[0]:
                line = self.text_gen.generate(pen.drawable)
                box = pen.draw(gc, x, y, line, color=conf.fg_color,
                               shadow_color=conf.shadow_color, shadow_width=conf.shadow_width)
                x += box.width + conf.line_spacing
        else:
            while y + conf.font_size + conf.line_spacing + margin < canvas.size[1]:
                line = self.text_gen.generate(pen.drawable)
                box = pen.draw(gc, x, y, line, color=conf.fg_color,
                               shadow_color=conf.shadow_color, shadow_width=conf.shadow_width)
                y += box.height + conf.line_spacing
        return canvas

    def collate_fn(batch):
        return batch

    def __getitem__(self, i):
        im = self.gen_text_block_image()
        return self.transforms(im)

    def __len__(self):
        return self.num_samples


def main():
    workers = cpu_count() - 2
    if workers <= 0:
        workers = 1
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--size", "-s", type=int, default=640,
                        help="output image size")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="output directory")
    parser.add_argument("--num-samples", "-n", type=int, required=True,
                        help="number of images to generate")
    parser.add_argument("--seed", type=int, default=71,
                        help="random seed")
    parser.add_argument("--postfix", type=str, help="filename postfix")
    parser.add_argument("--font-names", type=str, nargs="+", default=DEFAULT_FONT_NAMES,
                        help="font names to use")
    parser.add_argument("--font-dir", type=str, default=DEFAULT_FONT_DIR,
                        help="font dir")
    parser.add_argument("--bg-dir", type=str, help="background image directory")
    parser.add_argument("--num-workers", type=int, default=workers, help="number of worker process")

    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    postfix = "_" + args.postfix if args.postfix else ""

    img_gen = torch.utils.data.DataLoader(
        TextImageGenerator(args),
        collate_fn=TextImageGenerator.collate_fn,
        worker_init_fn=lambda worker_id: set_seed(worker_id + args.seed),
        batch_size=1,
        shuffle=False,
        num_workers=4, drop_last=False)

    for i, x in enumerate(tqdm(img_gen, ncols=80)):
        im = x[0]
        output_path = path.join(args.output_dir, f"__TEXT_{i}_{postfix}.png")
        im.save(output_path)


if __name__ == "__main__":
    main()
