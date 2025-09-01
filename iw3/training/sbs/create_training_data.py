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
from torchvision import transforms as T
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


OFFSET = 32


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0], (size, size))
    results = []
    for im in images:
        results.append(TF.crop(im, i, j, h, w))

    return tuple(results)


def random_hard_example_crop(size, n, *images):
    assert n > 0
    results = []
    for i in range(n):
        crop_images = random_crop(size, *images)
        # depth is last image
        depth = crop_images[-1]
        stdv = TF.to_tensor(depth).float().std(dim=[1, 2]).sum().item()
        results.append((stdv, crop_images))

    results = sorted(results, key=lambda v: v[0], reverse=True)
    results = [images for stdv, images in results]
    return results


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
    unpad_size = int(im_org.width * divergence * 0.01 + 2)

    im_org = TF.crop(im_org, 0, unpad_size, im_org.height, im_org.width - unpad_size * 2)
    im_depth = TF.crop(im_depth, 0, unpad_size, im_depth.height, im_depth.width - unpad_size * 2)
    im_l = TF.crop(im_l, 0, unpad_size, im_l.height, im_l.width - unpad_size * 2)
    im_r = TF.crop(im_r, 0, unpad_size, im_r.height, im_r.width - unpad_size * 2)
    im_mask_l = TF.crop(im_mask_l, 0, unpad_size, im_mask_l.height, im_mask_l.width - unpad_size * 2)
    im_mask_r = TF.crop(im_mask_r, 0, unpad_size, im_mask_r.height, im_mask_r.width - unpad_size * 2)
    assert im_org.size == im_l.size and im_org.size == im_r.size and im_org.size == im_depth.size == im_mask_r.size == im_mask_r.size
    if min(im_org.size) < size:
        return

    # im_l.save(filename_base + "_debug_l.png")
    # im_depth.save(filename_base + "_debug_d.png")
    images = (im_org, im_l, im_r, im_mask_l, im_mask_r, im_depth)
    names = ("_C", "_L", "_R", "_ML", "_MR", "_D")
    crops = random_hard_example_crop(size, num_samples * 3, *images)
    for seq, images in enumerate(crops[:num_samples]):
        for im, postfix in zip(images, names):
            output_filename = filename_base + "_" + str(seq) + postfix + ".png"
            im.save(output_filename, format="png", pnginfo=metadata)


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


def gen_divergence(width, divergence_level):
    if divergence_level == 1:
        # max divergence == 5
        # NOTE: min(32.0 / (width * 0.5) * 100, max_divergence) is correct but use this
        max_divergence = min(OFFSET / (width * 0.5) * 100, 5.0)
        if random.uniform(0, 1) < 0.7:
            return random.choice([2., 2.5, 3.0])
        else:
            return random.uniform(0., max_divergence)
    elif divergence_level == 2:
        # max divergence == 8
        max_divergence = min(OFFSET / (width * 0.5) * 100, 8.0)
        min_divergence = min(max_divergence - 0.1, 3.0)
        if random.uniform(0, 1) < 0.7:
            return random.choice([4.0, 5.0, 6.0])
        else:
            return random.uniform(min_divergence, max_divergence)
    elif divergence_level == 3:
        # max divergence == 11
        max_divergence = min(OFFSET / (width * 0.5) * 100, 11.0)
        min_divergence = min(max_divergence - 0.1, 6.0)
        if random.uniform(0, 1) < 0.7:
            return random.choice([7.0, 8.0, 9.0])
        else:
            return random.uniform(min_divergence, max_divergence)


def gen_convergence(weak_random_convergence):
    if weak_random_convergence:
        # weak random
        if random.uniform(0, 1) < 0.7:
            return 0.5
        else:
            return random.uniform(0.5 - 0.2, 0.5 + 0.2)
    else:
        # full random
        if random.uniform(0, 1) < 0.7:
            return random.choice([0.0, 0.5, 1.0])
        else:
            return random.uniform(0., 1.)


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
    if model_type.startswith("ZoeD"):
        return random.choice([0] * 6 + [1, 2, 3, 4])
    else:
        return random.choice([0] * 6 + [1, 2, 2, 2, 3, 4])


def gen_depth_aa(model_type):
    if model_type.startswith("Any_V2"):
        return random.choice([True, False])
    else:
        return False


def main(args):
    from ...depth_model_factory import create_depth_model

    assert args.min_size - OFFSET * 2 >= args.size
    # force_update_midas()
    # force_update_zoedepth()

    max_workers = cpu_count() // 2 or 1
    filename_prefix = f"{args.prefix}_{args.model_type}_" if args.prefix else args.model_type + "_"
    model = create_depth_model(args.model_type)
    model.load(gpu=args.gpu, resolution=args.resolution)
    model.disable_ema()

    if args.eval_only:
        target_dataset_type = ("eval",)
    else:
        target_dataset_type = ("eval", "train")

    for dataset_type in target_dataset_type:
        input_dir = path.join(args.dataset_dir, dataset_type)
        output_dir = path.join(args.data_dir, dataset_type)
        if not path.exists(input_dir):
            print(f"Error: `{input_dir}` not found", file=sys.stderr)
            return

    for dataset_type in target_dataset_type:
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
                    convergence = gen_convergence(args.weak_random_convergence)
                    edge_dilation = gen_edge_dilation(args.model_type)
                    depth_aa = gen_depth_aa(args.model_type)
                    flip_aug = False  # random.choice([True, False, False, False])
                    enable_amp = True

                    with torch.inference_mode():
                        if random.choice([True, False]):
                            # resize depth size to image size
                            im_s = random_resize(im, args.min_size, args.max_size)
                            divergence = gen_divergence(im_s.width, args.divergence_level)
                            depth = model.infer(im_s, tta=flip_aug, enable_amp=enable_amp,
                                                edge_dilation=edge_dilation, depth_aa=depth_aa)
                            depth = F.interpolate(depth.unsqueeze(0), (im_s.height, im_s.width),
                                                  mode="bilinear", align_corners=True, antialias=True).squeeze(0)
                        else:
                            # resize image size to depth size
                            depth = model.infer(im, tta=flip_aug, enable_amp=enable_amp,
                                                edge_dilation=edge_dilation, depth_aa=depth_aa)
                            im_s = F.interpolate(TF.to_tensor(im).unsqueeze(0), depth.shape[-2:],
                                                 mode="bilinear", align_corners=False, antialias=True).squeeze(0)
                            im_s = im_s.clamp(0, 1)
                            im_s = TF.to_pil_image(im_s)
                            divergence = gen_divergence(im_s.width, args.divergence_level)

                        assert im_s.height == depth.shape[-2] and im_s.width == depth.shape[-1]

                        depth = model.minmax_normalize_chw(depth)
                        np_depth16 = (depth * 0xffff).round().to(torch.uint16).squeeze(0).cpu().numpy()

                    c = TF.to_tensor(im_s).unsqueeze(0).to(depth.device)
                    depth_f = apply_mapper(depth, mapper).unsqueeze(0)
                    left_eye, right_eye, left_mask, right_mask = apply_divergence_forward_warp(
                        c, depth_f, divergence, convergence,
                        method="forward_fill",
                        synthetic_view="both",
                        return_mask=True,
                        inconsistent_shift=True)

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
    parser.add_argument("--divergence-level", type=int, default=1, choices=[1, 2, 3], help="divergence level. 1=0-5, 2=3-8, 3=6-11")
    parser.add_argument("--weak-random-convergence", action="store_true", help="Use weak random convergence. 0.3-0.7")
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
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval")

    parser.set_defaults(handler=main)

    return parser
