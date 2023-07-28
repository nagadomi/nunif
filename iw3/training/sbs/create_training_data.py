import os
import sys
import argparse
from os import path
from tqdm import tqdm
import random
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)
from nunif.utils.pil_io import load_image_simple
from PIL.PngImagePlugin import PngInfo
from nunif.utils.image_loader import ImageLoader, list_images
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count


def save_images(im_org, im_sbs, im_depth, divergence, convergence, filename_base, size, num_samples):
    im_l = TF.crop(im_sbs, 0, 0, im_org.height, im_org.width)
    im_r = TF.crop(im_sbs, 0, im_org.width, im_org.height, im_org.width)
    assert im_org.size == im_l.size and im_org.size == im_r.size and im_org.size == im_depth.size

    metadata = PngInfo()
    min_v, max_v = im_depth.getextrema()
    metadata.add_text("sbs_width", str(im_org.width))
    metadata.add_text("sbs_height", str(im_org.height))
    metadata.add_text("sbs_divergence", str(round(divergence, 6)))
    metadata.add_text("sbs_convergence", str(round(convergence, 6)))
    metadata.add_text("sbs_depth_max", str(max_v))
    metadata.add_text("sbs_depth_min", str(min_v))

    # im_sbs.save(filename_base + "_LRF.png")

    stride = size // 2
    w, h = im_org.size
    seq = 1
    rect_groups = []
    for y in range(0, h, stride):
        if not y + size <= h:
            break
        for x in range(0, w, stride):
            if not x + size <= w:
                break
            rects = []
            for im, postfix in zip((im_org, im_l, im_r, im_depth), ("_C", "_L", "_R", "_D")):
                rect = TF.crop(im, y, x, size, size)
                rects.append((filename_base + "_" + str(seq) + postfix + ".png", rect))
            rect_groups.append(rects)
            seq += 1
    if not rect_groups:
        return

    random.shuffle(rect_groups)
    for i in range(min(num_samples, len(rect_groups))):
        rects = rect_groups[i]
        for output_filename, rect in rects:
            rect.save(output_filename, format="png", pnginfo=metadata)


def random_resize(im, min_size, max_size):
    w, h = im.size
    common_width = [640, 768, 1024, 1280, 1440, 1920]
    common_height = [320, 480, 576, 720, 1080]
    common_width = [s for s in common_width if min_size <= s and s <= max_size and s <= w]
    common_height = [s for s in common_height if min_size <= s and s <= max_size and s <= h]

    if random.uniform(0, 1) < 0.5 and common_height and common_width:
        # random select common format
        if w > h:
            new_w = random.choice(common_width)
            if new_w != w:
                new_h = int(h * (new_w / w))
                interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
                im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)
        else:
            new_h = random.choice(common_height)
            if new_h != h:
                new_w = int(w * (new_h / h))
                interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
                im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)
        if im.width < min_size:
            pad_w1 = (min_size - im.width) // 2
            pad_w2 = (min_size - im.width) - pad_w1
            im = TF.pad(im, (pad_w1, 0, pad_w2, 0), padding_mode="constant")
        if im.height < min_size:
            pad_h1 = (min_size - im.height) // 2
            pad_h2 = (min_size - im.height) - pad_h1
            im = TF.pad(im, (0, pad_h1, 0, pad_h2), padding_mode="constant")
    else:
        # full random resize
        min_factor = min_size / min(im.size)
        max_factor = max_size / max(im.size)
        scale_factor = random.uniform(min_factor, max_factor)
        new_w = int(im.width * scale_factor)
        new_h = int(im.height * scale_factor)
        assert min(new_w, new_h) >= min_size and max(new_w, new_h) <= max_size
        interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
        im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)

    assert min_size <= min(im.size) and max(im.size) <= max_size

    return im


def gen_divergence():
    if random.uniform(0, 1) < 0.7:
        return random.choice([2., 2.5])
    else:
        return random.uniform(0., 2.5)


def gen_convergence():
    if random.uniform(0, 1) < 0.7:
        return random.choice([0.0, 0.5, 1.0])
    else:
        return random.uniform(0., 1.)


def main(args):
    import numba
    from .depthmap_utils import generate_sbs
    from ... import zoedepth_model as ZU
    # force_update_midas()
    # force_update_zoedepth()

    max_workers = cpu_count() // 2 or 1
    numba.set_num_threads(max_workers)

    filename_prefix = args.prefix + "_" if args.prefix else ""
    model = ZU.load_model(model_type="ZoeD_N", gpu=args.gpu, height=args.zoed_height)

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

        loader = ImageLoader(files=list_images(input_dir), max_queue_size=128,
                             load_func=load_image_simple,
                             load_func_kwargs={"color": "rgb"})
        seq = 1
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for im, meta in tqdm(loader, ncols=80):
                if im is None:
                    continue
                if min(im.size) < args.min_size:
                    continue
                for _ in range(args.times):
                    im_s = random_resize(im, args.min_size, args.max_size)
                    output_base = path.join(output_dir, filename_prefix + str(seq))
                    divergence = gen_divergence()
                    convergence = gen_convergence()
                    sbs, depth = generate_sbs(
                        model, im_s,
                        divergence=divergence, convergence=convergence,
                        flip_aug=random.choice([True, False]),
                        enable_amp=random.choice([True, False]))
                    f = pool.submit(save_images, im_s, sbs, depth,
                                    divergence, convergence,
                                    output_base, args.size, args.num_samples)
                    # f.result() # debug
                    futures.append(f)
                    seq += 1
                for f in futures:
                    f.result()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "sbs",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max-size", type=int, default=1920, help="max image size")
    parser.add_argument("--min-size", type=int, default=320, help="min image size")
    parser.add_argument("--prefix", type=str, default="", help="prefix for output filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--size", type=int, default=256, help="crop size")
    parser.add_argument("--times", type=int, default=4,
                        help="number of times an image is used for random scaling")
    parser.add_argument("--num-samples", type=int, default=8, help="max random crops")
    parser.add_argument("--zoed-height", type=int, help="input height for ZoeDepth model")

    parser.set_defaults(handler=main)

    return parser
