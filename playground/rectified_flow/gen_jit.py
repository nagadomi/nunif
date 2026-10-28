import argparse
import os
from os import path

import torch
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid

from nunif.device import autocast, create_device
from nunif.models import Model, load_model

from .celeba_dataset import IMAGE_SIZE
from .dit import DiT
from .dit_self_cond import DiTSelfCond
from .dit_without_cond import DiTWithoutCond
from .rf1_jit import sample


def load_rf_model(model: Model, model_path: str, device: torch.device) -> Model:
    if path.isdir(model_path):
        model_path = path.join(model_path, model.name + ".pth")
    model, _ = load_model(model_path, model=model)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=str, required=True, help="model directory or model file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="output image path")
    parser.add_argument("--num-samples", type=int, default=16, help="number of samples to generate")
    parser.add_argument("--steps", type=int, default=25, help="sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for sampling")
    parser.add_argument("--nrow", type=int, default=None, help="grid column count (default: min(num_samples, 8))")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id, -1 for cpu")
    parser.add_argument("--without-cond", action="store_true", help="without timestep conditioning")
    parser.add_argument(
        "--self-cond", action="store_true", help="use self conditioning instead of timestep conditioning"
    )
    args = parser.parse_args()

    model: Model
    if args.without_cond:
        model = DiTWithoutCond()
    elif args.self_cond:
        model = DiTSelfCond()
    else:
        model = DiT()

    device = create_device(args.gpu)
    model = load_rf_model(model, args.model_dir, device)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    all_outputs: list[torch.Tensor] = []
    for i in range(0, args.num_samples, args.batch_size):
        cur_batch = min(args.batch_size, args.num_samples - i)
        x = torch.randn((cur_batch, 3, IMAGE_SIZE, IMAGE_SIZE), generator=rng, device=device)
        with torch.inference_mode(), autocast(device):
            out = sample(model, x, steps=args.steps)
        all_outputs.append(out.cpu())

    outputs = torch.cat(all_outputs, dim=0)
    nrow = args.nrow if args.nrow is not None else min(args.num_samples, 8)
    grid = make_grid(outputs, nrow=nrow)

    output_dir = path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    TF.to_pil_image(grid).save(args.output)


if __name__ == "__main__":
    main()
