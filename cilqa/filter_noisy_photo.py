# Tool to filter low quality jpeg files
import os
from os import path
import argparse
import shutil
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import torch
from nunif.utils.image_loader import ImageLoader
# from .download_models import main as download_main
from nunif.utils import pil_io
from nunif.models import load_model, get_model_device
from nunif.logger import logger
from .utils import predict_grain_noise_psnr
from .models import grain_noise_level  # noqa


DEFAULT_CHECKPOINT_FILE = path.join(path.dirname(__file__), "pretrained_models", "grain_noise_level.pth")


def copyfile(src, dst, symlink):
    if symlink:
        if path.exists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
    else:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image directory")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_FILE, help="model parameter file")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--num-patches", type=int, default=8, help="number of 128x128 patches used per image")
    parser.add_argument("--psnr", type=int, default=40, help="quality threshold")
    parser.add_argument("--symlink", action="store_true",
                        help=("create symbolic links, "
                              "instead of copying the real files (recommended on linux)"))
    parser.add_argument("--score-prefix", action="store_true", help="add score prefix to the output filename")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    model, _ = load_model(args.checkpoint, device_ids=args.gpu)
    model.eval()
    loader = ImageLoader(args.input,
                         load_func=pil_io.load_image_simple,
                         load_func_kwargs={"color": "rgb"})

    with torch.no_grad(), PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        futures = []
        for im, meta in tqdm(loader, ncols=80):
            psnr = predict_grain_noise_psnr(model, im, num_patches=args.num_patches)
            logger.debug(f"{meta['filename']}: psnr: {round(psnr, 3)}")
            if psnr >= args.psnr:
                src = path.abspath(meta["filename"])
                if args.score_prefix:
                    prefix = f"{str(int(psnr)).zfill(2)}_"
                    dst = path.abspath(path.join(args.output, prefix + path.basename(src)))
                else:
                    dst = path.abspath(path.join(args.output, path.basename(src)))
                futures.append(pool.submit(copyfile, src, dst, args.symlink))

                if len(futures) > 100:
                    for f in futures:
                        f.result()
                    futures = []

        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
