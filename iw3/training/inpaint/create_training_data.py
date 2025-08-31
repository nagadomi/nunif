import os
import sys
import argparse
import torch
from os import path
from tqdm import tqdm
import random
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import ImageLoader, list_images
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
from iw3.utils import get_mapper
from iw3.forward_warp import nonwarp_mask


OFFSET = 32


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


def gen_convergence():
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
    return random.choice([0] * 6 + [1, 2, 2, 2, 3, 4])


def resize(im, min_size):
    w, h = im.size
    if w > h:
        new_h = min_size
        new_w = int(w * (new_h / h))
    else:
        new_w = min_size
        new_h = int(h * (new_w / w))

    interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
    im = TF.resize(im, (new_h, new_w), interpolation=interpolation, antialias=True)

    return im


def apply_mapper(depth, mapper):
    return get_mapper(mapper)(depth)


def gen_data(im, model, args):
    mapper = gen_mapper(model.is_metric())
    convergence = gen_convergence()
    edge_dilation = gen_edge_dilation(args.model_type)
    image_size = random.choices([720, 1080], weights=[0.25, 0.5], k=1)[0]

    with torch.inference_mode():
        im = resize(im, image_size)
        divergence = gen_divergence(im.width, args.divergence_level)
        depth_aa = random.choice([False, False, False, True])
        depth = model.infer(im, edge_dilation=edge_dilation, depth_aa=depth_aa, tta=False, enable_amp=True)
        depth = model.minmax_normalize_chw(depth)
        c = TF.to_tensor(im).unsqueeze(0).to(depth.device)
        depth = apply_mapper(depth, mapper).unsqueeze(0)

        _, mask, overlap_mask = nonwarp_mask(c, depth, divergence=divergence, convergence=convergence)

        c_flip = torch.flip(c, (-1,))
        depth_flip = torch.flip(depth, (-1,))

        _, mask_flip, overlap_mask_flip = nonwarp_mask(c_flip, depth_flip, divergence=divergence, convergence=convergence)

        # NOTE: mask_closing() is applied on the model or dataset side.

        return [(c[0], mask.float()[0], overlap_mask.float()[0]),
                (c_flip[0], mask_flip.float()[0], overlap_mask_flip.float()[0])]


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0], (size, size))
    results = []
    for im in images:
        results.append(TF.crop(im, i, j, h, w))

    return tuple(results)


def random_hard_example_crop(size, n, *images):
    assert n > 0
    results = []
    for _ in range(n):
        crop_images = random_crop(size, *images)
        mask = crop_images[-1]
        mask_sum = mask.float().sum(dim=[1, 2]).sum().item()
        results.append((mask_sum, crop_images))

    results = sorted(results, key=lambda v: v[0], reverse=True)
    results = [images for stdv, images in results]
    return results[0]


def save_images(c, mask, overlap_mask, output_base, size, num_samples):
    for i in range(num_samples):
        rgb_rect, overlap_mask_rect, mask_rect = random_hard_example_crop(size, 4, c, overlap_mask, mask)
        TF.to_pil_image(rgb_rect).save(f"{output_base}_{i}_C.png")
        TF.to_pil_image(mask_rect).save(f"{output_base}_{i}_M.png")
        TF.to_pil_image(overlap_mask_rect).save(f"{output_base}_{i}_O.png")


def main(args):
    from ...depth_model_factory import create_depth_model

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
            seq = 0
            for im, meta in tqdm(loader, ncols=80):
                if im is None:
                    continue

                data = gen_data(im, model, args)
                for c, mask, overlap_mask in data:
                    seq += 1
                    output_base = path.join(output_dir, filename_prefix + str(seq))
                    f = pool.submit(save_images, c, mask, overlap_mask, output_base, args.size, args.num_samples)
                    # f.result()  # debug
                    futures.append(f)

            for f in futures:
                f.result()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "inpaint",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--divergence-level", type=int, default=1, choices=[1, 2, 3],
                        help="divergence level. 1=0-5, 2=3-8, 3=6-11")
    parser.add_argument("--prefix", type=str, default="", help="prefix for output filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--size", type=int, default=512, help="crop size")
    parser.add_argument("--num-samples", type=int, default=4, help="max random crops")
    parser.add_argument("--resolution", type=int, help="input resolution for depth model")
    parser.add_argument("--model-type", type=str, default="ZoeD_Any_L", help="depth model")
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval")

    parser.set_defaults(handler=main)

    return parser
