import argparse
import os
from os import path
from tqdm import tqdm
import torch
from nunif.device import create_device, autocast
import torchvision.transforms.functional as TF
from nunif.models import load_model
from .dataset import SODDataset
from ... import models  # noqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True, help="model file")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids. -1 for CPU")
    parser.add_argument("--output", "-o", type=str, required=True, help="output file/directory")
    parser.add_argument("--data-dir", type=str, required=True, help="dataset dir")
    args = parser.parse_args()

    device = create_device(args.gpu[0])

    model, _ = load_model(args.model_file, device_ids=args.gpu)
    model.eval()

    depth_dir = path.join(args.data_dir, "DUTS-TE", "depth")
    mask_dir = path.join(args.data_dir, "DUTS-TE", "DUTS-TE-Mask")
    rgb_dir = path.join(args.data_dir, "DUTS-TE", "DUTS-TE-Image")
    dataset = SODDataset(depth_dir=depth_dir,
                         mask_dir=mask_dir,
                         rgb_dir=rgb_dir,
                         size=192,
                         training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    os.makedirs(args.output, exist_ok=True)
    i = 0
    for x, *_ in tqdm(loader, ncols=80):
        x = x.to(device)
        with torch.inference_mode(), autocast(x.device):
            batch = model(x)
            batch = (batch > 0.5).float()
        for im in batch:
            TF.to_pil_image(im).save(path.join(args.output, f"{i}.png"))
            i += 1


if __name__ == "__main__":
    main()
