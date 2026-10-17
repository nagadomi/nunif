import argparse
import copy
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
from os import path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from iw3.depth_model_factory import create_depth_model
from iw3.stereo_model_factory import create_stereo_model
from iw3.utils import apply_divergence, create_parser, postprocess_image
from nunif.utils.image_loader import ImageLoader, list_images
from nunif.utils.pil_io import load_image_simple


def gen_divergence() -> float:
    if random.uniform(0, 1) < 0.7:
        return random.uniform(1.0, 5.0)
    else:
        return random.uniform(5.0, 16.0)


def gen_convergence() -> float:
    return random.uniform(0.0, 1.0)


def gen_mapper(is_metric: bool) -> str:
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


def resize_max_len_pil(im: Image.Image, max_size: int) -> Image.Image:
    w, h = im.size
    if max(w, h) > max_size:
        if w >= h:
            new_w = max_size
            new_h = max(1, int(round(h * (max_size / w))))
        else:
            new_h = max_size
            new_w = max(1, int(round(w * (max_size / h))))
        im = TF.resize(im, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
    return im


def resize_to_max_len_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if w >= h:
        new_w = size
        new_h = max(1, int(round(h * (size / w))))
    else:
        new_h = size
        new_w = max(1, int(round(w * (size / h))))
    x = TF.resize(x, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
    return torch.clamp(x, 0.0, 1.0)


def save_image_tensor(tensor: torch.Tensor, output_path: str) -> None:
    im = TF.to_pil_image(tensor.cpu())
    im.save(output_path, format="png")


def main(args: argparse.Namespace) -> None:
    model = create_depth_model(args.model_type)
    model.load(gpu=args.gpu, resolution=args.resolution)
    model.disable_ema()

    target_dataset_type = ("eval",) if args.eval_only else ("eval", "train")

    for dataset_type in target_dataset_type:
        input_dir = path.join(args.dataset_dir, dataset_type)
        if not path.exists(input_dir):
            print(f"Error: `{input_dir}` not found", file=sys.stderr)
            return

    stereo_formats = [
        ("full_SBS", {"half_sbs": False, "tb": False, "half_tb": False}),
        ("half_SBS", {"half_sbs": True, "tb": False, "half_tb": False}),
        ("full_TB", {"half_sbs": False, "tb": True, "half_tb": False}),
        ("half_TB", {"half_sbs": False, "tb": False, "half_tb": True}),
    ]

    iw3_parser = create_parser(required_true=False)
    base_args = iw3_parser.parse_args([])
    base_args.state = {"convergence_model": None}
    side_model = create_stereo_model("monobw", divergence=5.0, device_id=args.gpu)

    for dataset_type in target_dataset_type:
        print(f"** {dataset_type}")
        input_dir = path.join(args.dataset_dir, dataset_type)
        output_dir = path.join(args.data_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)

        loader = ImageLoader(
            files=list_images(input_dir),
            max_queue_size=128,
            load_func=load_image_simple,
            load_func_kwargs={"color": "rgb"},
        )
        seq = 1
        with PoolExecutor(max_workers=args.max_workers) as pool:
            futures = []
            for im, _meta in tqdm(loader, ncols=80):
                if im is None:
                    continue

                im = resize_max_len_pil(im, args.max_size)

                with torch.inference_mode():
                    depth = model.infer(im, enable_amp=True)
                    assert depth is not None
                    if depth.shape[-2:] != (im.height, im.width):
                        depth = F.interpolate(
                            depth.unsqueeze(0),
                            (im.height, im.width),
                            mode="bilinear",
                            align_corners=True,
                            antialias=True,
                        ).squeeze(0)
                    depth = model.minmax_normalize_chw(depth)

                c = TF.to_tensor(im).to(depth.device)

                flat_tensor = resize_to_max_len_tensor(c, args.size)
                flat_path = path.join(output_dir, f"{seq:06d}_FLAT.png")
                futures.append(pool.submit(save_image_tensor, flat_tensor, flat_path))

                for format_name, format_args in stereo_formats:
                    divergence = gen_divergence()
                    convergence = gen_convergence()
                    mapper = gen_mapper(model.is_metric()) if args.mapper == "random" else args.mapper
                    if divergence < 2.5:
                        synthetic_view = random.choice(["both", "right", "left"])
                    else:
                        synthetic_view = random.choice(["both", "right", "left"])

                    divergence_args = copy.copy(base_args)
                    divergence_args.divergence = divergence
                    divergence_args.convergence = convergence
                    divergence_args.mapper = mapper
                    divergence_args.method = "monobw"
                    divergence_args.synthetic_view = synthetic_view
                    divergence_args.half_sbs = format_args["half_sbs"]
                    divergence_args.half_tb = format_args["half_tb"]
                    divergence_args.tb = format_args["tb"]

                    with torch.inference_mode():
                        left_eye, right_eye = apply_divergence(depth, c, divergence_args, side_model=side_model)
                        assert left_eye is not None and right_eye is not None
                        sbs = postprocess_image(left_eye, right_eye, divergence_args)
                        sbs = resize_to_max_len_tensor(sbs, args.size)

                    out_path = path.join(output_dir, f"{seq:06d}_{format_name}.png")
                    futures.append(pool.submit(save_image_tensor, sbs, out_path))

                seq += 1

            for f in futures:
                f.result()


def register(
    subparsers: argparse._SubParsersAction, default_parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    max_workers = cpu_count() // 2 or 1
    parser = subparsers.add_parser(
        "iw3.format", parents=[default_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--model-type", type=str, default="Any_V3_Mono", help="depth model")
    parser.add_argument("--resolution", type=int, help="input resolution for depth model")
    parser.add_argument("--size", type=int, default=128, help="output max size")
    parser.add_argument("--max-size", type=int, default=1024, help="pre-process max image size")
    parser.add_argument("--mapper", type=str, default="random", help="depth mapper function or 'random'")
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval")
    parser.add_argument(
        "--max-workers", type=int, default=max_workers, help="Number of worker threads. Set to 1 to disable."
    )

    parser.set_defaults(handler=main)

    return parser
