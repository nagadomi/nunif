import argparse
import os
from tqdm import tqdm
import torch
from nunif.device import create_device, autocast
import torchvision.transforms.functional as TF
from nunif.models import load_model
from ... import models  # noqa
from ...depth_model_factory import create_depth_model
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--model-file", type=str, required=True, help="SOD model file (pth)")
    parser.add_argument("--depth-model", type=str, default="Any_V2_S", help="depth model")
    parser.add_argument("--resolution", type=int, help="depth resolution !Not SOD resolution!")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids. -1 for CPU")
    args = parser.parse_args()

    if args.depth_model.startswith("VDA_"):
        raise ValueError("VDA is not supported")

    device = create_device(args.gpu[0])
    sod_model, _ = load_model(args.model_file, device_ids=args.gpu)
    sod_model = sod_model.eval().fuse()
    depth_model = create_depth_model(args.depth_model)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()

    loader = ImageLoader(directory=args.input, load_func=load_image_simple, load_func_kwargs={"color": "rgb"})
    os.makedirs(args.output, exist_ok=True)
    for im, meta in tqdm(loader, ncols=80):
        if not im:
            continue
        rgb = TF.to_tensor(im).to(device)
        with torch.inference_mode(), autocast(device):
            depth = depth_model.infer(rgb, edge_dilation=0, tta=False, enable_amp=True)
            depth = depth_model.minmax_normalize_chw(depth)
            sal, *_ = sod_model.infer(rgb.unsqueeze(0), depth.unsqueeze(0))
            mask = (sal > 0.5).float().squeeze(0)

            rgb = TF.resize(rgb, size=mask.shape[-2:])
            vis = (rgb * 0.3 + mask * 0.7).clamp(0, 1)
            TF.to_pil_image(vis).save(
                os.path.join(args.output, os.path.splitext(os.path.basename(meta["filename"]))[0] + ".png")
            )


if __name__ == "__main__":
    main()
