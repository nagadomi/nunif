# visualize discriminator patch output
# python -m waifu2x.training.visualize_discriminator_output -i tmp/sr_output -o ./tmp/discmap --model ./models/noise3_scale4x_discriminator.pth
import argparse
import os
import torch
from tqdm import tqdm
from os import path
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple, to_tensor
from nunif.models import load_model
from nunif.device import autocast
from nunif.modules.pad import get_pad_size
import waifu2x.models
import waifu2x.models.discriminator # noqa
from nunif.modules.permute import window_partition2d, window_reverse2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input file or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--model", type=str, required=True, help="discriminator pth path")
    parser.add_argument("--tile", type=int, default=128, help="tile size")
    args = parser.parse_args()

    model = load_model(args.model)[0]
    model = model.cuda().eval()

    os.makedirs(args.output, exist_ok=True)

    if path.isfile(args.input):
        files = [args.input]
    else:
        files = ImageLoader.listdir(args.input)

    for fn in tqdm(files, ncols=80):
        x, _ = load_image_simple(fn, color="rgb")
        x = to_tensor(x).unsqueeze(0)
        pad = get_pad_size(x, args.tile)
        x = F.pad(x, pad, mode="reflect").cuda()

        with autocast(x.device), torch.no_grad():
            output_shape = x.shape
            x = window_partition2d(x, args.tile)
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            outputs = []

            for i in range(x.shape[0]):
                xx = x[i:i + 1]
                # cond should be GT by design, but it has been simplified.
                cond = xx
                z = model(xx, cond)
                if isinstance(z, (list, tuple)):
                    # use first output only
                    z = z[0]
                z = F.interpolate(z, size=xx.shape[-2:], mode="nearest")
                z = (z.clamp(-2, 2) + 2) / 4.0
                xx = xx.mean(dim=1, keepdim=True)
                xx = torch.cat([xx * 0.25, z, z], dim=1)[0]
                # xx = torch.cat([z, z, z], dim=1)[0]
                outputs.append(xx.cpu())

            outputs = torch.stack(outputs)
            outputs = outputs.reshape(B, N, C, H, W)
            x = window_reverse2d(outputs, output_shape, args.tile)[0]
            TF.to_pil_image(x).save(path.join(args.output, path.basename(fn)))


if __name__ == "__main__":
    main()
