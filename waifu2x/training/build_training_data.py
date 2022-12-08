import os
from os import path
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from tqdm import tqdm
import torchvision.transforms.functional as TF
from nunif.utils.pil_io import load_image_simple
from nunif.utils.image_loader import ImageLoader


def split_image(filepath_prefix, im, size, stride, reject_rate):
    w, h = im.size
    rects = []
    for y in range(0, h, stride):
        if not y + size <= h:
            break
        for x in range(0, w, stride):
            if not x + size <= w:
                break
            rect = TF.crop(im, y, x, size, size)
            color_stdv = TF.to_tensor(rect).permute(1, 2, 0).view(-1, 3).std(dim=0).sum().item()
            rects.append((rect, color_stdv))

    n_reject = int(len(rects) * reject_rate)
    rects = [v[0] for v in sorted(rects, key=lambda v: v[1], reverse=True)][0:len(rects) - n_reject]

    index = 0
    for rect in rects:
        rect.save(f"{filepath_prefix}_{index}.png")
        rect.close()
        index += 1

    im.close()

    return None


def main(args):
    def wait_pool(futures):
        for f in futures:
            f.result()
        return []

    for dataset_type in ("validation", "train"):
        print(f"** {dataset_type}")
        input_dir = path.join(args.dataset_dir, dataset_type)
        output_dir = path.join(args.data_dir, dataset_type)

        os.makedirs(output_dir, exist_ok=True)

        loader = ImageLoader(
            directory=input_dir,
            load_func=load_image_simple,
            load_func_kwargs={"color": "rgb"})
        index = 0
        futures = []
        with PoolExecutor() as pool:
            for im, _ in tqdm(loader, ncols=80):
                if im is None:
                    continue
                f = pool.submit(split_image,
                                path.join(output_dir, str(index)),
                                im, args.size, int(args.size * args.stride), args.reject_rate)
                futures.append(f)
                index += 1

            for f in tqdm(futures, ncols=80):
                f.result()


def register(subparsers):
    parser = subparsers.add_parser("waifu2x")
    parser.add_argument("--size", type=int, default=640,
                        help="image size")
    parser.add_argument("--stride", type=float, default=0.25,
                        help="stride_size = int(size * stride)")
    parser.add_argument("--reject-rate", type=float, default=0.5,
                        help="reject rate for hard example mining")
    parser.set_defaults(handler=main)
    return parser
