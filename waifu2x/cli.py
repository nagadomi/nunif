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


DEFAULT_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "cunet", "art"))


def convert_files(ctx, files, args, enable_amp):
    loader = ImageLoader(files=files, max_queue_size=128,
                         load_func=IL.load_image,
                         load_func_kwargs={"color": "rgb", "keep_alpha": True})
    os.makedirs(args.output, exist_ok=True)
    futures = []
    with torch.no_grad(), PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        for im, meta in tqdm(loader, ncols=60):
            rgb, alpha = IL.to_tensor(im, return_alpha=True)
            rgb, alpha = ctx.convert(
                rgb, alpha, args.method, args.noise_level,
                args.tile_size, args.batch_size,
                args.tta, enable_amp=enable_amp)
            output_filename = set_image_ext(path.basename(meta["filename"]), format=args.format)
            if args.depth is not None:
                meta["depth"] = args.depth
            futures.append(pool.submit(
                IL.save_image,
                IL.to_image(rgb, alpha),
                filename=path.join(args.output, output_filename),
                meta=meta,
                format=args.format))
        for f in futures:
            f.result()


def convert_file(ctx, args, enable_amp):
    _, ext = path.splitext(args.output)
    fmt = ext.lower()[1:]
    if fmt not in {"png", "webp", "jpeg", "jpg"}:
        raise ValueError(f"Unable to recognize image extension: {fmt}")

    with torch.no_grad():
        im, meta = IL.load_image(args.input, color="rgb", keep_alpha=True)
        rgb, alpha = IL.to_tensor(im, return_alpha=True)
        rgb, alpha = ctx.convert(rgb, alpha, args.method, args.noise_level,
                                 args.tile_size, args.batch_size,
                                 args.tta, enable_amp=enable_amp)
        if args.depth is not None:
            meta["depth"] = args.depth
        IL.save_image(IL.to_image(rgb, alpha),
                      filename=args.output, meta=meta,
                      format=fmt)


def load_files(txt):
    files = []
    with open(txt, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
    return files


def main(args):
    ctx = Waifu2x(model_dir=args.model_dir, gpus=args.gpu)
    ctx.load_model(args.method, args.noise_level)

    if path.isdir(args.input):
        convert_files(ctx, ImageLoader.listdir(args.input), args, enable_amp=args.amp)
    else:
        if path.splitext(args.input)[-1] in (".txt", ".csv"):
            convert_files(ctx, load_files(args.input), args, enable_amp=args.amp)
        else:
            convert_file(ctx, args, enable_amp=args.amp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="model dir")
    parser.add_argument("--noise-level", "-n", type=int, default=0, choices=[0, 1, 2, 3], help="noise level")
    parser.add_argument("--method", "-m", type=str, choices=["scale", "noise", "noise_scale"], default="noise_scale", help="method")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=True, help="output file or directory")
    parser.add_argument("--input", "-i", type=str, required=True, help="input file or directory. (*.txt, *.csv) for image list")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--amp", action="store_true", help="with half float")
    parser.add_argument("--image-lib", type=str, choices=["pil", "wand"], default="pil",
                        help="image library to encode/decode images")
    parser.add_argument("--depth", type=int, help="bit-depth of output image. enabled only with `--image-lib wand`")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "webp", "jpeg"], help="output image format")
    args = parser.parse_args()
    logger.debug(f"waifu2x.cli.main: {str(args)}")
    if args.image_lib == "wand":
        from nunif.utils import wand_io as IL
    else:
        from nunif.utils import pil_io as IL

    main(args)
