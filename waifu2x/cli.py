# waifu2x
import os
from os import path
import torch
import argparse
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from nunif.logger import logger
from nunif.utils import load_image, save_image, ImageLoader
from .waifu2x import Waifu2x
from .models import CUNet, VGG7, UpConv7


DEFAULT_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "cunet", "art"))


def convert_files(ctx, files, args, enable_amp):
    loader = ImageLoader(files=files, max_queue_size=128)
    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad(), PoolExecutor() as pool:
        for im, meta in tqdm(loader, ncols=60):
            z = ctx.convert(
                im, meta, args.method, args.noise_level,
                args.tile_size, args.batch_size,
                args.tta, enable_amp=enable_amp)
            output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            pool.submit(save_image, z, meta, path.join(args.output, output_filename))


def convert_file(ctx, args, enable_amp):
    with torch.no_grad():
        im, meta = load_image(args.input)
        z = ctx.convert(im, meta, args.method, args.noise_level,
                        args.tile_size, args.batch_size,
                        args.tta, enable_amp=enable_amp)
        save_image(z, meta, args.output)


def load_files(txt):
    files = []
    with open(txt, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
    return files


def main():
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
    args = parser.parse_args()
    logger.debug(f"waifu2x.cli.main: {str(args)}")

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
    main()
