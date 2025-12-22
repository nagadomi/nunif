import os
import argparse
from os import path
from tqdm import tqdm
import random
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import ImageLoader, list_images
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
from iw3.base_depth_model import BaseDepthModel


def main(args):
    from ...depth_model_factory import create_depth_model

    if args.model_type.startswith("VDA_"):
        raise ValueError("VDA is not supported")

    max_workers = cpu_count() // 2 or 1
    # use "_" * 3 deliminator for dataset
    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()

    # args.dataset_dir=DUTS-TR-Image
    if path.basename(args.dataset_dir) != "DUTS-TR-Image" and path.basename(args.dataset_dir) != "DUTS-TE-Image":
        raise ValueError("Use DUTS-TR-Image/DUTS-TE-Image for --dataset-dir")

    input_dir = args.dataset_dir
    # args.data_dir=aug/{prefix_dir}/
    if path.dirname(args.data_dir) == path.dirname(input_dir):
        ValueError(f"Use {path.dirname(input_dir)} for --data-dir")
    if args.resolution is not None:
        output_dir = path.join(args.data_dir, "depth", f"{args.model_type}_{args.resolution}")
    else:
        output_dir = path.join(args.data_dir, "depth", args.model_type)
    os.makedirs(output_dir, exist_ok=True)

    loader = ImageLoader(files=list_images(input_dir),
                         load_func=load_image_simple,
                         load_func_kwargs={"color": "rgb"})
    with PoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for im, meta in tqdm(loader, ncols=80):
            if im is None:
                continue
            edge_dilation = random.randint(0, 2)
            depth_aa = random.choice([True, False, False, False])
            depth = depth_model.infer(im, tta=False, enable_amp=True,
                                      edge_dilation=edge_dilation, depth_aa=depth_aa)
            depth = depth_model.minmax_normalize_chw(depth)

            output_filename = path.join(output_dir, path.splitext(path.basename(meta["filename"]))[0] + ".png")
            f = pool.submit(BaseDepthModel.save_normalized_depth, depth, output_filename)
            # f.result()  # debug
            futures.append(f)

        for f in futures:
            f.result()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "dsod",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--resolution", type=int, help="input resolution for depth model")
    parser.add_argument("--model-type", type=str, default="ZoeD_Any_L", help="depth model")
    parser.set_defaults(handler=main)

    return parser
