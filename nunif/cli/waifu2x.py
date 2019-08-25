# waifu2x
import os
from os import path
import torch
import argparse
from .. logger import logger
from .. utils import load_image, save_image, ImageLoader
from .. tasks.waifu2x import Waifu2x
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PoolExecutor


DEFAULT_MODEL_DIR = path.abspath(path.join(path.dirname(path.abspath(__file__)),
                                 "..", "..", "pretrained_models", "waifu2x", "cunet", "art"))


def convert_dir(ctx, args):
    loader = ImageLoader(directory=args.input, max_queue_size=128)
    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad(), PoolExecutor() as pool:
        for im, meta in tqdm(loader, ncols=60):
            z = ctx.convert(im, meta, args.method, args.noise_level, args.tile_size, args.batch_size, args.tta)
            output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            pool.submit(save_image, z, meta, path.join(args.output, output_filename))


def convert_file(ctx, args):
    with torch.no_grad():
        im, meta = load_image(args.input)
        z = ctx.convert(im, meta, args.method, args.noise_level, args.tile_size, args.batch_size, args.tta)
        save_image(z, meta, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="model dir")
    parser.add_argument("--noise-level", "-n", type=int, default=1, choices=[0, 1, 2, 3], help="noise level")
    parser.add_argument("--method", "-m", type=str, choices=["scale", "noise", "noise_scale"], default="noise_scale", help="method")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=True, help="output")
    parser.add_argument("--input", "-i", type=str, required=True, help="input")
    parser.add_argument("--tta", action="store_true", help="use tta")
    args = parser.parse_args()
    logger.debug(str(args))

    ctx = Waifu2x(model_dir=args.model_dir, gpus=args.gpu)
    ctx.load_model(args.method, args.noise_level)

    if path.isdir(args.input):
        convert_dir(ctx, args)
    else:
        convert_file(ctx, args)
