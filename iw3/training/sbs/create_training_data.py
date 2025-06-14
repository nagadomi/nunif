# when unicode error, set LANG=C
import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
from os import path
from tqdm import tqdm
import random
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)
import torch.nn.functional as F
from nunif.utils.pil_io import load_image_simple
from PIL.PngImagePlugin import PngInfo
from nunif.utils.image_loader import ImageLoader, list_images
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
from iw3.utils import get_mapper
from iw3.forward_warp import apply_divergence_forward_warp
from .stereoimage_generation import create_stereoimages


OFFSET = 32


def save_images(im_org, im_sbs, im_mask_sbs, im_depth, divergence, convergence, mapper, filename_base, size, num_samples):
    im_l = TF.crop(im_sbs, 0, 0, im_org.height, im_org.width)
    im_r = TF.crop(im_sbs, 0, im_org.width, im_org.height, im_org.width)
    im_mask_l = TF.crop(im_mask_sbs, 0, 0, im_org.height, im_org.width)
    im_mask_r = TF.crop(im_mask_sbs, 0, im_org.width, im_org.height, im_org.width)

    assert im_org.size == im_l.size and im_org.size == im_r.size and im_org.size == im_depth.size
    assert im_org.size == im_mask_l.size and im_org.size == im_mask_r.size

    metadata = PngInfo()
    min_v, max_v = im_depth.getextrema()
    metadata.add_text("sbs_width", str(im_org.width))
    metadata.add_text("sbs_height", str(im_org.height))
    metadata.add_text("sbs_divergence", str(round(divergence, 6)))
    metadata.add_text("sbs_convergence", str(round(convergence, 6)))
    metadata.add_text("sbs_depth_max", str(max_v))
    metadata.add_text("sbs_depth_min", str(min_v))
    metadata.add_text("sbs_mapper", mapper)

    # remove replication padding area
    unpad_size = int(im_org.width * divergence * 0.5 * 0.01 + 2)

    im_org = TF.crop(im_org, 0, unpad_size, im_org.height, im_org.width - unpad_size * 2)
    im_depth = TF.crop(im_depth, 0, unpad_size, im_depth.height, im_depth.width - unpad_size * 2)
    im_l = TF.crop(im_l, 0, unpad_size, im_l.height, im_l.width - unpad_size * 2)
    im_r = TF.crop(im_r, 0, unpad_size, im_r.height, im_r.width - unpad_size * 2)
    im_mask_l = TF.crop(im_mask_l, 0, unpad_size, im_mask_l.height, im_mask_l.width - unpad_size * 2)
    im_mask_r = TF.crop(im_mask_r, 0, unpad_size, im_mask_r.height, im_mask_r.width - unpad_size * 2)
    assert im_org.size == im_l.size and im_org.size == im_r.size and im_org.size == im_depth.size == im_mask_r.size == im_mask_r.size

    # im_l.save(filename_base + "_debug_l.png")
    # im_depth.save(filename_base + "_debug_d.png")

    stride = size // 4
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
            for im, postfix in zip((im_org, im_l, im_r, im_mask_l, im_mask_r, im_depth), ("_C", "_L", "_R", "_ML", "_MR", "_D")):
                rect = TF.crop(im, y, x, size, size)

                rects.append((filename_base + "_" + str(seq) + postfix + ".png", rect))
            rect_groups.append(rects)
            seq += 1
    if not rect_groups:
        return
    if len(rect_groups) > num_samples:
        if random.uniform(0, 1) < 0.5:
            # sorted by depth std
            rects_sorted = sorted([(rects, TF.to_tensor(rects[3][1]).float().std(dim=[1, 2]).sum().item())
                                   for rects in rect_groups],
                                  key=lambda d: d[1], reverse=True)
            rect_groups = [d[0] for d in rects_sorted[:num_samples]]
        else:
            random.shuffle(rect_groups)

    for i in range(min(num_samples, len(rect_groups))):
        rects = rect_groups[i]
        for output_filename, rect in rects:
            rect.save(output_filename, format="png", pnginfo=metadata)


def random_resize(im, min_size, max_size):
    w, h = im.size
    common_size = [384, 392, 512, 518] * 3 + [768, 770, 1036, 1024]
    if random.uniform(0, 1) < 0.75:
        # random select common format
        if w < h:
            new_w = random.choice(common_size)
            if new_w < w:
                new_h = int(h * (new_w / w))
                interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
                im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)
        else:
            new_h = random.choice(common_size)
            if new_h < h:
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
        if new_w < w:
            if not (min(new_w, new_h) >= min_size and max(new_w, new_h) <= max_size):
                new_h = max(min_size, new_h)
                new_w = max(min_size, new_w)
                new_h = min(max_size, new_h)
                new_w = min(max_size, new_w)
            interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
            im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)

    if max_size < im.width:
        im = TF.crop(im, 0, 0, im.height, max_size)
    if max_size < im.height:
        im = TF.crop(im, 0, 0, max_size, im.width)

    # assert min_size <= min(im.size) and max(im.size) <= max_size

    return im


def apply_mapper(depth, mapper):
    return get_mapper(mapper)(depth)


def gen_divergence(width, large_divergence):
    if not large_divergence:
        # max divergence == 5
        # NOTE: min(32.0 / (width * 0.5) * 100, max_divergence) is correct but use this
        max_divergence = min(OFFSET / width * 100, 5.0)
        if random.uniform(0, 1) < 0.7:
            return random.choice([2., 2.5, 3.0])
        else:
            return random.uniform(0., max_divergence)
    else:
        # max divergence == 10
        max_divergence = min(OFFSET / (width * 0.6) * 100, 10.0)
        min_divergence = min(max_divergence - 0.1, 3.0)
        if random.uniform(0, 1) < 0.7:
            return random.choice([5.0, 6.0, 7.0, 8.0])
        else:
            return random.uniform(min_divergence, max_divergence)


def gen_convergence(full_random_convergence):
    if full_random_convergence:
        if random.uniform(0, 1) < 0.7:
            return random.choice([0.0, 0.5, 1.0])
        else:
            return random.uniform(0., 1.)
    else:
        # weak random
        # needed to prevent strange distortion of grid_sample
        if random.uniform(0, 1) < 0.7:
            return 0.5
        else:
            return random.uniform(0.5 - 0.125, 0.5 + 0.125)


def gen_mapper(is_metric):
    if is_metric:
        if random.uniform(0, 1) < 0.7:
            return "div_6"
        else:
            return random.choice(["none", "div_25", "div_10", "div_4", "div_2", "div_1"])
    else:
        if random.uniform(0, 1) < 0.7:
            return "none"
        else:
            return random.choice(["inv_mul_1", "inv_mul_2", "inv_mul_3", "mul_1", "mul_2", "mul_3"])


def gen_edge_dilation(model_type):
    return random.choice([2] * 6 + [0, 1, 3, 4])


def gen_depth_aa(model_type):
    if model_type.startswith("Any_V2"):
        return random.choice([True, False])
    else:
        return False


def main(args):
    import numba
    from ...depth_model_factory import create_depth_model

    assert args.min_size - OFFSET * 2 >= args.size
    # force_update_midas()
    # force_update_zoedepth()

    max_workers = cpu_count() // 2 or 1
    numba.set_num_threads(max_workers)

    filename_prefix = f"{args.prefix}_{args.model_type}_" if args.prefix else args.model_type + "_"
    model = create_depth_model(args.model_type)
    model.load(gpu=args.gpu, resolution=args.resolution)
    model.disable_ema()

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
                    mapper = gen_mapper(model.is_metric()) if args.mapper == "random" else args.mapper
                    output_base = path.join(output_dir, filename_prefix + str(seq))
                    convergence = gen_convergence(args.full_random_convergence)
                    edge_dilation = gen_edge_dilation(args.model_type)
                    depth_aa = gen_depth_aa(args.model_type)
                    flip_aug = False  # random.choice([True, False, False, False])
                    enable_amp = True

                    with torch.inference_mode():
                        if random.choice([True, False]):
                            # resize depth size to image size
                            im_s = random_resize(im, args.min_size, args.max_size)
                            divergence = gen_divergence(im_s.width, args.large_divergence)
                            depth = model.infer(im_s, tta=flip_aug, enable_amp=enable_amp,
                                                edge_dilation=edge_dilation, depth_aa=depth_aa)
                            depth = F.interpolate(depth.unsqueeze(0), (im_s.height, im_s.width),
                                                  mode="bilinear", align_corners=True, antialias=True).squeeze(0)
                        else:
                            # resize image size to depth size
                            depth = model.infer(im, tta=flip_aug, enable_amp=enable_amp,
                                                edge_dilation=edge_dilation, depth_aa=depth_aa)
                            im_s = TF.resize(im, depth.shape[-2:], InterpolationMode.BILINEAR, antialias=True)
                            divergence = gen_divergence(im_s.width, args.large_divergence)

                        assert im_s.height == depth.shape[-2] and im_s.width == depth.shape[-1]

                        depth = model.minmax_normalize_chw(depth)
                        np_depth16 = (depth * 0xffff).to(torch.uint16).squeeze(0).cpu().numpy()

                    if args.method == "polylines":
                        np_depth_f = apply_mapper(depth, mapper).squeeze(0).numpy().astype(np.float64)
                        sbs = create_stereoimages(
                            np.array(im_s, dtype=np.uint8),
                            np_depth_f,
                            divergence, modes=["left-right"],
                            convergence=convergence)[0]
                    elif args.method == "forward_fill":
                        c = TF.to_tensor(im_s).unsqueeze(0).to(depth.device)
                        depth_f = apply_mapper(depth, mapper).unsqueeze(0)
                        left_eye, right_eye, left_mask, right_mask = apply_divergence_forward_warp(c, depth_f, divergence, convergence,
                                                                                                   method="forward_fill", synthetic_view="both",
                                                                                                   return_mask=True)

                        sbs = torch.cat([left_eye, right_eye], dim=3).squeeze(0)
                        sbs = TF.to_pil_image(torch.clamp(sbs, 0., 1.))
                        mask_sbs = torch.cat([left_mask, right_mask], dim=3).squeeze(0).float()
                        mask_sbs = TF.to_pil_image(mask_sbs)

                    f = pool.submit(save_images, im_s, sbs, mask_sbs, Image.fromarray(np_depth16),
                                    divergence, convergence,
                                    mapper,
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

    parser.add_argument("--max-size", type=int, default=920, help="max image size")
    parser.add_argument("--large-divergence", action="store_true", help="Use divergence up to 10 instead of 5")
    parser.add_argument("--full-random-convergence", action="store_true", help="Use full random convergence")
    parser.add_argument("--min-size", type=int, default=320, help="min image size")
    parser.add_argument("--prefix", type=str, default="", help="prefix for output filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--size", type=int, default=256, help="crop size")
    parser.add_argument("--times", type=int, default=4,
                        help="number of times an image is used for random scaling")
    parser.add_argument("--num-samples", type=int, default=2, help="max random crops")
    parser.add_argument("--resolution", type=int, help="input resolution for depth model")
    parser.add_argument("--model-type", type=str, default="ZoeD_N", help="depth model")
    parser.add_argument("--mapper", type=str, default="random", help="depth mapper function")
    parser.add_argument("--method", type=str, default="forward_fill", choices=["forward_fill", "polylines"], help="divergence method")

    parser.set_defaults(handler=main)

    return parser
