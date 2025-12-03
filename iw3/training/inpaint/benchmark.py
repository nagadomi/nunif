import argparse
from os import path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import iw3.models  # noqa
from nunif.models import load_model
from nunif.modules.lpips import LPIPSMetric
from nunif.device import create_device, autocast
from .dataset import InpaintDataset
from .dataset_video import VideoInpaintDataset


def psnr(input, target, mask=None):
    if mask is not None:
        mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1), reduction="none")
        mse = (mse * mask).sum() / mask.sum()
        return 10 * torch.log10(1.0 / (mse + 1.0e-6))
    else:
        mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
        return 10 * torch.log10(1.0 / (mse + 1.0e-6))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", "-i", type=str, required=True, help="dataset dir. <dataset_dir>/eval is used")
    parser.add_argument("--checkpoint-file", type=str, required=True, help="model file")
    parser.add_argument("--video", action="store_true", help="Use video dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="device ids; if -1 is specified, use CPU")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg"],
                        help="LPIPS base model")

    args = parser.parse_args()
    device = create_device(args.gpu)
    model = load_model(args.checkpoint_file)[0].eval().to(device)
    if args.video:
        dataset_class = VideoInpaintDataset
        dataset_kwargs = dict(model_sequence_offset=0)
    else:
        dataset_class = InpaintDataset
        dataset_kwargs = {}

    model_offset = model.i2i_offset
    dataset = dataset_class(path.join(args.data_dir, "eval"), model_offset=model_offset, training=False, **dataset_kwargs)
    lpips = LPIPSMetric(net=args.lpips_net).to(device)

    lpips_sum = 0
    psnr_sum = 0
    lpips_mask_sum = 0
    psnr_mask_sum = 0
    data_count = 0

    for x, mask, y, *_ in tqdm(dataset, ncols=80):
        mask_y = F.pad(mask, (-model_offset,) * 4).float()
        if mask_y.sum() == 0:
            continue

        if x.ndim == 3:
            # image
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask_y = mask_y.unsqueeze(0)

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask_y = mask_y.to(device)

        with torch.inference_mode(), autocast(device):
            x, mask = model.preprocess(x, mask)
            z = model(x, mask)

            lpips_sum = lpips_sum + lpips(z, y).item()
            lpips_mask_sum = lpips_mask_sum + lpips(z, y, mask=mask_y).item()
            psnr_sum = psnr_sum + psnr(z, y).item()
            psnr_mask_sum = psnr_mask_sum + psnr(z, y, mask=mask_y).item()
            data_count += 1

    print("* Image")
    print(f"PSNR↑: {round(psnr_sum / data_count, 4)}, LPIPS↓: {round(lpips_sum / data_count, 4)}")
    print("* Mask Region")
    print(f"PSNR↑: {round(psnr_mask_sum / data_count, 4)}, LPIPS↓: {round(lpips_mask_sum / data_count, 4)}")


if __name__ == "__main__":
    main()
