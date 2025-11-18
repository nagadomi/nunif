import os
import sys
import argparse
import torch
from os import path
from tqdm import tqdm
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import ImageLoader, list_images
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
from iw3.base_depth_model import BaseDepthModel


def save_images(
        filename_base,
        depth_da2, min_value_da2, max_value_da2,
        depth_da3, min_value_da3, max_value_da3,
):
    assert depth_da2.shape == depth_da3.shape

    for model_type, depth, min_value, max_value in zip(
            ("DA2", "DA3"),
            (depth_da2, depth_da3),
            (min_value_da2, min_value_da3),
            (max_value_da2, max_value_da3),
    ):
        output_filename = f"{filename_base}_{model_type}.png"
        BaseDepthModel.save_normalized_depth(
            depth, output_filename,
            min_depth_value=min_value, max_depth_value=max_value
        )


def normalize(depth):
    min_value = depth.min()
    max_value = depth.max()

    depth = (depth - min_value) / (max_value - min_value)
    return depth, min_value, max_value


def main(args):
    from ...depth_model_factory import create_depth_model

    max_workers = cpu_count() // 2 or 1
    filename_prefix = f"{args.prefix}_{args.resolution}" if args.prefix else str(args.resolution) + "_"
    model_da2 = create_depth_model("Any_V2_L")
    model_da2.load(gpu=args.gpu, resolution=args.resolution)
    model_da2.disable_ema()
    model_da3 = create_depth_model("Any_V3_Mono")
    model_da3.load(gpu=args.gpu, resolution=args.resolution, raw_output=True)
    model_da3.disable_ema()

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

                output_base = path.join(output_dir, filename_prefix + str(seq))
                enable_amp = True

                with torch.inference_mode():
                    depth_da2 = model_da2.infer(im, tta=False, enable_amp=enable_amp,
                                                edge_dilation=0, depth_aa=False)
                    depth_da3 = model_da3.infer(im, tta=False, enable_amp=enable_amp,
                                                edge_dilation=0, depth_aa=False)
                    depth_da2, min_value_da2, max_value_da2 = normalize(-depth_da2)
                    depth_da3, min_value_da3, max_value_da3 = normalize(depth_da3)

                f = pool.submit(save_images, output_base,
                                depth_da2, min_value_da2, max_value_da2,
                                depth_da3, min_value_da3, max_value_da3)

                # f.result() # debug
                futures.append(f)
                seq += 1
            for f in futures:
                f.result()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "iw3.da3mono",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--resolution", type=int, default=392, help="resolution. 392, 504, 518")
    parser.add_argument("--prefix", type=str, default="", help="prefix for output filename")

    parser.set_defaults(handler=main)

    return parser
