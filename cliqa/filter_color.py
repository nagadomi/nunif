# Tool to filter color/grayscale images
import os
from os import path
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import torch
from nunif.logger import logger
from .utils import create_patch_loader, copyfile
from nunif.device import create_device


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image directory")
    parser.add_argument("--num-patches", type=int, default=8, help="number of 128x128 patches used per image")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--threshold", type=int, default=0.04, help="RGB stdv threshold")
    parser.add_argument("--invert", action="store_true", help="extract color images")
    parser.add_argument("--symlink", action="store_true",
                        help=("create symbolic links, "
                              "instead of copying the real files (recommended on linux)"))

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = create_device(args.gpu[0])
    loader = create_patch_loader(args.input, num_patches=args.num_patches, num_workers=cpu_count())
    with torch.inference_mode(), PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        futures = []
        for x, filename in tqdm(loader, ncols=80):
            x, filename = x[0], filename[0]
            if not filename:  # load error
                continue
            stdv = torch.std(x.to(device), dim=1, keepdim=True).max().item()
            logger.debug(f"{filename}: stdv: {round(stdv, 4)}")
            if (not args.invert and stdv <= args.threshold) or (args.invert and stdv > args.threshold):
                src = path.abspath(filename)
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
