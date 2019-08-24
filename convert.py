import os
from os import path
import torch
import torchvision.transforms.functional as TF
import argparse
from nunif.utils import tiled_render, make_alpha_border, load_model, ImageLoader
from tqdm import tqdm


def to_tensor(im):
    im = TF.to_tensor(im)
    return im.view(1, im.shape[0], im.shape[1], im.shape[2])


def convert_dir(model, device, args):
    loader = ImageLoader(directory=args.input, max_queue_size=256)
    os.makedirs(args.output, exist_ok=True)
    for im, meta in loader:
        rgb = TF.to_tensor(im)
        if "alpha" in meta:
            alpha = TF.to_tensor(meta["alpha"])
            rgb = make_alpha_border(rgb.to(device), alpha.to(device), model.offset).to("cpu")
        z = tiled_render(rgb, model, device, tile_size=args.tile_size, batch_size=args.batch_size)
        output_filename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
        TF.to_pil_image(z).save(path.join(args.output, output_filename))


def convert_file(model, device, args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True, help="model file")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="gpu ids")
    parser.add_argument("--num-workers", type=int, default=8, help="number of worker threads")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=True, help="output")
    parser.add_argument("--input", "-i", type=str, required=True, help="input")
    args = parser.parse_args()

    if args.gpu[0] < 0:
        device = 'cpu'
    else:
        device = 'cuda:{}'.format(args.gpu[0])
    model = load_model(args.model_file).to(device)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    if path.isdir(args.input):
        convert_dir(model, device, args)
    else:
        convert_file(model, device, args)
