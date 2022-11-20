# image to image
import os
from os import path
import torch
import torchvision.transforms.functional as TF
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from .. utils import tiled_render, simple_render, ImageLoader
from .. models import load_model, get_model_config, I2IBaseModel
from .. logger import logger
from .. addon import load_addons


def save_image(im, output):
    im.save(output)


def make_loader(input_path):
    if path.isdir(input_path):
        loader = ImageLoader(directory=input_path, max_queue_size=256)
        is_dir = True
    else:
        loader = ImageLoader(files=[input_path], max_queue_size=256)
        is_dir = False
    return loader, is_dir


def convert_with_tiled_render(model, args):
    loader, is_dir = make_loader(args.input)
    in_size = get_model_config(model, "i2i_in_size")
    in_grayscale = get_model_config(model, "i2i_in_channels") == 1
    tile_size = in_size if in_size is not None else args.tile_size

    if is_dir:
        os.makedirs(args.output, exist_ok=True)
    with torch.no_grad(), PoolExecutor() as pool:
        for im, meta in tqdm(loader, ncols=60):
            if in_grayscale:
                im = TF.to_grayscale(im)
            z = tiled_render(TF.to_tensor(im), model, tile_size=tile_size, batch_size=args.batch_size).to("cpu")
            if is_dir:
                output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
                pool.submit(save_image, TF.to_pil_image(z), path.join(args.output, output_filename))
            else:
                pool.submit(save_image, TF.to_pil_image(z), args.output)


def convert_with_simple_render_single(model, args):
    loader, is_dir = make_loader(args.input)
    in_grayscale = get_model_config(model, "i2i_in_channels") == 1

    if is_dir:
        os.makedirs(args.output, exist_ok=True)
    with torch.no_grad(), PoolExecutor() as pool:
        for im, meta in tqdm(loader, ncols=60):
            if in_grayscale:
                im = TF.to_grayscale(im)
            z = simple_render(TF.to_tensor(im), model).to('cpu')
            if is_dir:
                output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
                pool.submit(save_image, TF.to_pil_image(z), path.join(args.output, output_filename))
            else:
                pool.submit(save_image, TF.to_pil_image(z), args.output)


def convert_with_simple_render_batch(model, args):
    # TODO: not test
    loader, is_dir = make_loader(args.input)
    in_size = get_model_config(model, "i2i_in_size")
    in_channels = get_model_config(model, "i2i_in_channels")
    if is_dir:
        os.makedirs(args.output, exist_ok=True)

    with torch.no_grad(), PoolExecutor() as pool:
        output_paths = [None] * args.batch_size
        minibatch = torch.zeros((args.batch_size, in_channels,
                                 in_size, in_size))
        minibatch_index = 0
        for im, meta in tqdm(loader, ncols=60):
            x = im
            if in_channels == 1:
                x = TF.to_grayscale(x)
            if in_size is not None:
                h = w = in_size
                if not (h == x.height and w == x.width):
                    x = TF.resize(x, (h, w))
            minibatch[minibatch_index] = TF.to_tensor(x)
            if is_dir:
                output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
                output_paths[minibatch_index] = path.join(args.output, output_filename)
            else:
                output_paths[minibatch_index] = args.output
            minibatch_index += 1
            if minibatch_index == minibatch.shape[0]:
                z = simple_render(minibatch, model).to('cpu')
                for i in range(minibatch_index):
                    pool.submit(save_image, TF.to_pil_image(z[i]), output_paths[i])
                minibatch_index = 0
        if minibatch_index > 0:
            z = simple_render(minibatch[0:minibatch_index], model).to('cpu')
            for i in range(minibatch_index):
                pool.submit(save_image, TF.to_pil_image(z[i]), output_paths[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True, help="model file")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tiled-render", "-t", action='store_true', help="use tiled render")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--tta", action='store_true', help="use TTA")
    parser.add_argument("--output", "-o", type=str, required=True, help="output file/directory")
    parser.add_argument("--input", "-i", type=str, required=True, help="input file/directory")
    parser.add_argument("--addon", type=str, nargs="+", help="dependent addons")
    args = parser.parse_args()
    logger.debug(str(args))

    load_addons(args.addon)

    model, _ = load_model(args.model_file, device_ids=args.gpu)
    if not isinstance(model, I2IBaseModel):
        raise ValueError("The model not a subclass of I2IBaseModel")
    model.eval()

    if args.tiled_render:
        convert_with_tiled_render(model, args)
    else:
        if get_model_config(model, "i2i_in_size") is None:
            # variable input size
            convert_with_simple_render_single(model, args)
        else:
            convert_with_simple_render_batch(model, args)


if __name__ == "__main__":
    main()
