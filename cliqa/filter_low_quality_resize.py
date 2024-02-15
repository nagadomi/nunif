# Tool to filter resized(upscaled) images
import os
from os import path
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import torch
from nunif.models import load_model
from nunif.logger import logger
from .utils import predict_resize_quality, create_patch_loader, copyfile
from .models import scale_factor  # noqa


DEFAULT_CHECKPOINT_FILE = path.join(path.dirname(__file__), "pretrained_models", "scale_factor.pth")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image directory")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_FILE, help="model parameter file")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--num-patches", type=int, default=8, help="number of 128x128 patches used per image")
    parser.add_argument("--quality", type=int, default=95, help="quality threshold")
    parser.add_argument("--invert", action="store_true", help="extract low quality resized images")
    parser.add_argument("--symlink", action="store_true",
                        help=("create symbolic links, "
                              "instead of copying the real files (recommended on linux)"))
    parser.add_argument("--score-prefix", action="store_true", help="add score prefix to the output filename")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    model, _ = load_model(args.checkpoint, device_ids=args.gpu, weights_only=True)
    model.eval()
    loader = create_patch_loader(args.input, num_patches=args.num_patches, num_workers=cpu_count())
    with torch.inference_mode(), PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        futures = []
        for x, filename in tqdm(loader, ncols=80):
            x, filename = x[0], filename[0]
            if not filename:  # load error
                continue
            quality = predict_resize_quality(model, x)
            logger.debug(f"{filename}: psnr: {round(quality, 3)}")
            if (not args.invert and quality >= args.quality) or (args.invert and quality < args.quality):
                src = path.abspath(filename)
                if args.score_prefix:
                    prefix = f"{str(int(quality)).zfill(3)}_"
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
    from .download_models import main as download_main

    download_main()
    main()
