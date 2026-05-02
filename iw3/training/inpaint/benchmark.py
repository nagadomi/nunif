import argparse
import os
from os import path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import iw3.models  # noqa
from nunif.models import load_model
from nunif.modules.lpips import LPIPSMetric
from nunif.device import create_device, autocast
from .dataset import InpaintDataset
from .dataset_video import VideoInpaintDataset


def psnr(input, target, mask=None):
    assert input.ndim == 4
    sum_psnr = 0
    input = input.clamp(0, 1)
    target = target.clamp(0, 1)
    if mask is not None:
        for x, y, m in zip(input, target, mask):
            assert m.sum() > 0
            mse = F.mse_loss(x, y, reduction="none")
            mse = (mse * m).sum() / m.sum()
            sum_psnr = sum_psnr + 10 * torch.log10(1.0 / (mse + 1.0e-6))
    else:
        for x, y in zip(input, target):
            mse = F.mse_loss(x, y)
            sum_psnr = sum_psnr + 10 * torch.log10(1.0 / (mse + 1.0e-6))

    return sum_psnr / input.shape[0]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", "-i", type=str, required=True, help="dataset dir. <dataset_dir>/eval is used")
    parser.add_argument("--checkpoint-file", type=str, required=True, help="model file")
    parser.add_argument("--video", action="store_true", help="Use video dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="device ids; if -1 is specified, use CPU")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg"],
                        help="LPIPS base model")
    parser.add_argument("--overlap-frames", type=int, nargs="+", default=[0],
                        help="overlap/reference frames <pre> <post> for video inpain")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory")

    args = parser.parse_args()
    if len(args.overlap_frames) == 1:
        overlap_frames = (args.overlap_frames[0], args.overlap_frames[0])
    elif len(args.overlap_frames) == 2:
        overlap_frames = (args.overlap_frames[0], args.overlap_frames[1])
    else:
        raise ValueError("--overlap-frames requires 1 or 2 values")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

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
    target_frames = 0
    processed_frames = 0
    skipped_frames = 0

    for x, mask, y, data_index in tqdm(dataset, ncols=80):
        target_frames += mask.shape[0]
        mask_y = F.pad(mask, (-model_offset,) * 4).float()
        if not mask_y.sum() > 0:
            skipped_frames += mask_y.shape[0]
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
            if args.video and overlap_frames:
                slice_start = overlap_frames[0] if overlap_frames[0] else None
                slice_end = -overlap_frames[1] if overlap_frames[1] else None
                s = slice(slice_start, slice_end)
                x = x[s]
                mask = mask[s]
                z = z[s]
                y = y[s]
                mask_y = mask_y[s]

            if args.output_dir:
                if args.video:
                    data_name = path.basename(dataset.folders[data_index])
                    output_dir = path.join(args.output_dir, data_name)
                    os.makedirs(output_dir, exist_ok=True)
                    for i in range(z.shape[0]):
                        TF.to_pil_image(z[i]).save(path.join(output_dir, f"{i}_z.png"))
                        TF.to_pil_image(y[i]).save(path.join(output_dir, f"{i}_y.png"))
                else:
                    data_name = path.splitext(path.basename(dataset.files[data_index]))[0]
                    TF.to_pil_image(z[0]).save(path.join(args.output_dir, f"{data_name}_z.png"))
                    TF.to_pil_image(y[0]).save(path.join(args.output_dir, f"{data_name}_y.png"))

            valid_index = (mask_y.sum(dim=(1, 2, 3)) > 0).nonzero(as_tuple=True)[0]
            z = z[valid_index]
            y = y[valid_index]
            mask_y = mask_y[valid_index]
            num_frames = z.shape[0]
            skipped_frames += x.shape[0] - mask_y.shape[0]
            if num_frames > 0:
                lpips_sum = lpips_sum + lpips(z, y).item() * num_frames
                lpips_mask_sum = lpips_mask_sum + lpips(z, y, mask=mask_y).item() * num_frames
                psnr_sum = psnr_sum + psnr(z, y).item() * num_frames
                psnr_mask_sum = psnr_mask_sum + psnr(z, y, mask=mask_y).item() * num_frames
                processed_frames += num_frames

    print("* Image")
    print(f"PSNR↑: {round(psnr_sum / processed_frames, 2)}, LPIPS↓: {round(lpips_sum / processed_frames, 4)}")
    print("* Mask Region")
    print(f"PSNR↑: {round(psnr_mask_sum / processed_frames, 2)}, LPIPS↓: {round(lpips_mask_sum / processed_frames, 4)}")
    print(f"\nTarget frames: {target_frames}, Processed frames: {processed_frames}, Skipped frames: {skipped_frames}")


if __name__ == "__main__":
    main()
