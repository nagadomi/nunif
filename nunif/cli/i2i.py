# image to image
import os
from os import path
import torch
import torchvision.transforms.functional as TF
import argparse
from nunif.utils import tiled_render, simple_render, load_model, load_image, ImageLoader, logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PoolExecutor


def convert(x, model, device, args):
    if model.in_channels == 1:
        x = TF.to_grayscale(x)

    if args.tiled_render:
        tile_size = args.tile_size if model.input_size is None else model.input_size
        z = tiled_render(TF.to_tensor(x), model, device, tile_size=tile_size, batch_size=args.batch_size)
    else:
        if model.input_size is not None:
            h = w = model.input_size
            if not (h == x.height and w == x.width):
                x = TF.resize(x, (h, w))
        z = simple_render(TF.to_tensor(x), model, device)
    return z


def save_image(im, output):
    im.save(output)


def convert_dir_tiled(model, device, args):
    loader = ImageLoader(directory=args.input, max_queue_size=256)
    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad(), PoolExecutor() as pool:
        for im, meta in tqdm(loader, ncols=60):
            z = convert(im, model, device, args)
            output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            pool.submit(save_image, TF.to_pil_image(z), path.join(args.output, output_filename))


def convert_dir_batch(model, device, args):
    if model.input_size is None and args.batch_size != 1:
        raise ValueError("model.input_size is None. use --tiled-render or --batch-size 1.")
    os.makedirs(args.output, exist_ok=True)
    loader = ImageLoader(directory=args.input, max_queue_size=256)
    with torch.no_grad(), PoolExecutor() as pool:
        output_paths = [None] * args.batch_size
        minibatch = torch.zeros((args.batch_size, model.in_channels,
                                 model.input_size, model.input_size))
        minibatch_index = 0
        for im, meta in tqdm(loader, ncols=60):
            x = im
            if model.in_channels == 1:
                x = TF.to_grayscale(x)
            h = w = model.input_size
            if not (h == x.height and w == x.width):
                x = TF.resize(x, (h, w))
            minibatch[minibatch_index] = TF.to_tensor(x)
            output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            output_paths[minibatch_index] = path.join(args.output, output_filename)
            minibatch_index += 1
            if minibatch_index == minibatch.shape[0]:
                z = simple_render(minibatch, model, device)
                for i in range(minibatch_index):
                    pool.submit(save_image, TF.to_pil_image(z[i]), output_paths[i])
                minibatch_index = 0
        if minibatch_index > 0:
            z = simple_render(minibatch[0:minibatch_index], model, device)
            for i in range(minibatch_index):
                pool.submit(save_image, TF.to_pil_image(z[i]), output_paths[i])


def convert_file(model, device, args):
    im, meta = load_image(args.input)
    with torch.no_grad():
        z = convert(im, model, device, args)
        TF.to_pil_image(z).save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True, help="model file")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tiled-render", "-t", action='store_true', help="use tiled render")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=True, help="output")
    parser.add_argument("--input", "-i", type=str, required=True, help="input")
    args = parser.parse_args()

    logger.debug(str(args))
    if args.gpu[0] < 0:
        device = 'cpu'
    else:
        device = 'cuda:{}'.format(args.gpu[0])
    model = load_model(args.model_file).to(device)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model.eval()

    if path.isdir(args.input):
        if args.tiled_render or args.batch_size == 1:
            convert_dir_tiled(model, device, args)
        else:
            convert_dir_batch(model, device, args)
    else:
        convert_file(model, device, args)
