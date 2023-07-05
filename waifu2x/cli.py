# waifu2x
import os
from os import path
import torch
import argparse
import csv
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from nunif.logger import logger
from nunif.utils.image_loader import ImageLoader
from nunif.utils.filename import set_image_ext
from .utils import Waifu2x
from .download_models import main as download_main


DEFAULT_ART_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "swin_unet", "art"))

DEFAULT_ART_SCAN_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "swin_unet", "art_scan"))

DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "swin_unet", "photo"))


def find_subdir(dirname):
    subdirs = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subdirs):
        subdirs.extend(find_subdir(dirname))
    return subdirs


def process_image(ctx, im, meta, args):
    rgb, alpha = IL.to_tensor(im, return_alpha=True)
    rgb, alpha = ctx.convert(
        rgb, alpha, args.method, args.noise_level,
        args.tile_size, args.batch_size,
        args.tta, enable_amp=not args.disable_amp)
    if args.depth is not None:
        meta["depth"] = args.depth
    depth = meta["depth"] if "depth" in meta and meta["depth"] is not None else 8
    if args.grayscale:
        meta["grayscale"] = True

    return IL.to_image(rgb, alpha, depth=depth)


def convert_files(ctx, files, output_dir, args):
    loader = ImageLoader(files=files, max_queue_size=128,
                         load_func=IL.load_image,
                         load_func_kwargs={"color": "rgb", "keep_alpha": True})
    os.makedirs(output_dir, exist_ok=True)
    futures = []
    with torch.inference_mode(), PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        for im, meta in tqdm(loader, ncols=60):
            output = process_image(ctx, im, meta, args)
            output_filename = set_image_ext(path.basename(meta["filename"]), format=args.format)
            futures.append(pool.submit(
                IL.save_image, output,
                filename=path.join(output_dir, output_filename),
                meta=meta, format=args.format))
        for f in futures:
            f.result()


def convert_file(ctx, args):
    _, ext = path.splitext(args.output)
    fmt = ext.lower()[1:]
    if fmt not in {"png", "webp", "jpeg", "jpg"}:
        raise ValueError(f"Unable to recognize image extension: {fmt}")

    with torch.inference_mode():
        im, meta = IL.load_image(args.input, color="rgb", keep_alpha=True)
        output = process_image(ctx, im, meta, args)
        IL.save_image(output, filename=args.output, meta=meta, format=fmt)


def load_files(txt):
    files = []
    with open(txt, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
    return files


def main(args):
    if args.model_dir is None:
        if args.style == "photo":
            model_dir = DEFAULT_PHOTO_MODEL_DIR
        elif args.style in {"scan", "art_scan"}:
            model_dir = DEFAULT_ART_SCAN_MODEL_DIR
        else:
            model_dir = DEFAULT_ART_MODEL_DIR
    else:
        model_dir = args.model_dir

    ctx = Waifu2x(model_dir=model_dir, gpus=args.gpu)
    ctx.load_model(args.method, args.noise_level)

    if path.isdir(args.input):
        if args.recursive:
            subdirs = sorted([args.input] + find_subdir(args.input))
            for input_dir in subdirs:
                files = ImageLoader.listdir(input_dir)
                if not files:
                    continue
                print(f"* {input_dir}")
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                convert_files(ctx, files, output_dir, args)
        else:
            convert_files(ctx, ImageLoader.listdir(args.input), args.output, args)
    else:
        if path.splitext(args.input)[-1] in (".txt", ".csv"):
            convert_files(ctx, load_files(args.input), args.output, args)
        else:
            convert_file(ctx, args)


if __name__ == "__main__":
    default_gpu = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=str, help="model dir")
    parser.add_argument("--noise-level", "-n", type=int, default=0, choices=[0, 1, 2, 3], help="noise level")
    parser.add_argument("--method", "-m", type=str,
                        choices=["scale4x", "scale2x",
                                 "noise_scale4x", "noise_scale2x",
                                 "scale", "noise", "noise_scale"],
                        default="noise_scale", help="method")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output file or directory")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file or directory. (*.txt, *.csv) for image list")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--image-lib", type=str, choices=["pil", "wand"], default="pil",
                        help="image library to encode/decode images")
    parser.add_argument("--depth", type=int,
                        help="bit-depth of output image. enabled only with `--image-lib wand`")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "webp", "jpeg"],
                        help="output image format")
    parser.add_argument("--style", type=str, choices=["art", "photo", "scan", "art_scan"],
                        help=("style for default model (art/scan/photo). "
                              "Ignored when --model-dir option is specified."))
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert to grayscale format")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="process all subdirectories")

    args = parser.parse_args()
    logger.debug(f"waifu2x.cli.main: {str(args)}")
    if args.image_lib == "wand":
        from nunif.utils import wand_io as IL
    else:
        from nunif.utils import pil_io as IL

    # alias for typo
    if args.method == "scale2x":
        args.method = "scale"
    elif args.method == "noise_scale2x":
        args.method = "noise_scale"

    # download models
    pretrained_model_dir = path.join(path.dirname(__file__), "pretrained_models")
    if not path.exists(pretrained_model_dir):
        download_main()

    # main
    main(args)
