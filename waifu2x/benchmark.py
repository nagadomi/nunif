import argparse
import csv
import math
import os
import time
import warnings
from os import path

import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm

import nunif.transforms.image_magick as IM
from nunif.device import autocast, create_device, device_is_cuda
from nunif.logger import logger
from nunif.modules.flat_color_loss import get_flat_color_mask
from nunif.transforms import functional as NF
from nunif.utils.image_loader import ImageLoader
from nunif.utils.render import tiled_render

try:
    from transformers import Swin2SRForImageSuperResolution

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


def load_files(txt):
    files = []
    with open(txt, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
    return files


def MSE(x1, x2, min_mse=1, mask=None):
    ignore = False
    if mask is None:
        mse = (x1 - x2).pow_(2).mean().clamp(min=min_mse).item()
    else:
        mask = mask.expand_as(x1)
        diff = (x1[mask] - x2[mask]).pow_(2)
        if diff.shape[0] == 0:
            mse = min_mse
            ignore = True
        else:
            mse = diff.mean().clamp(min=min_mse).item()

    return mse, ignore


def MSE2PSNR(mse):
    return 10 * math.log10((255 * 255) / mse)


def shift_jpeg_block(x, y, x_shift):
    if x_shift > 0:
        y_scale = y.shape[1] / x.shape[1]
        assert y_scale in {1, 2, 4}
        y_scale = int(y_scale)
        x_h, x_w = x.shape[1:]
        y_h, y_w = y.shape[1:]
        y_shift = x_shift * y_scale
        x = TF.crop(x, x_shift, x_shift, x_h - x_shift, x_w - x_shift)
        y = TF.crop(y, y_shift, y_shift, y_h - y_shift, y_w - y_shift)
        assert y.shape[1] == x.shape[1] * y_scale and y.shape[2] == x.shape[2] * y_scale

    return x, y


def add_jpeg_noise(x, y, args):
    if args.jpeg_yuv420:
        sampling_factor = IM.YUV420
    else:
        sampling_factor = IM.YUV444
    for i in range(args.jpeg_times):
        quality = args.jpeg_quality - i * args.jpeg_quality_down
        x = IM.jpeg_noise(x, sampling_factor, quality)
        if i != args.jpeg_times - 1:
            x, y = shift_jpeg_block(x, y, args.jpeg_shift)
    return x, y


def torch_resize(x, scale_factor, filter_type):
    if filter_type == "torch_bilinear":
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        ).squeeze(0)
    elif filter_type == "torch_bilinear_antialias":
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    elif filter_type == "torch_bilinear_align_corners":
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=True,
            antialias=False,
        ).squeeze(0)
    else:
        raise ValueError(filter_type)


def resize(x, scale_factor, filter_type, blur):
    if filter_type.startswith("torch_"):
        return torch_resize(x, scale_factor, filter_type)
    else:
        return IM.scale(x, scale_factor, filter_type=filter_type, blur=blur)


def make_input_waifu2x(gt, args):
    x = gt

    if args.method == "scale":
        x = resize(x, 0.5, filter_type=args.filter, blur=args.blur)
    elif args.method == "scale4x":
        x = resize(x, 0.25, filter_type=args.filter, blur=args.blur)
    elif args.method == "noise":
        x, gt = add_jpeg_noise(x, gt, args)
    elif args.method == "noise_scale":
        x, gt = add_jpeg_noise(resize(x, 0.5, filter_type=args.filter, blur=args.blur), gt, args)
    elif args.method == "noise_scale4x":
        x, gt = add_jpeg_noise(resize(x, 0.25, filter_type=args.filter, blur=args.blur), gt, args)
    return x, gt


def remove_border(x, border):
    return NF.crop(x, border, border, x.shape[1] - border, x.shape[2] - border)


def psnr256(x1, x2, color, flat_only):
    assert color in ("rgb", "y", "y_matlab")
    assert x1.shape == x2.shape
    if flat_only:
        # x1 = GT...
        mask = get_flat_color_mask(x1.unsqueeze(0)).squeeze(0).to(torch.bool)
    else:
        mask = None

    if color == "rgb":
        mse, ignore = MSE(NF.quantize256_f(x1), NF.quantize256_f(x2), mask=mask)
        psnr = MSE2PSNR(mse)
        return psnr, mse, ignore
    elif color == "y":
        mse, ignore = MSE(NF.quantize256_f(NF.rgb2y(x1)), NF.quantize256_f(NF.rgb2y(x2)), mask=mask)
        psnr = MSE2PSNR(mse)
        return psnr, mse, ignore
    elif color == "y_matlab":
        mse, ignore = MSE(NF.rgb2y_matlab(x1).float(), NF.rgb2y_matlab(x2).float(), mask=mask)
        psnr = MSE2PSNR(mse)
        return psnr, mse, ignore


SWIN2SR_MODELS = {
    "classical-sr-x2": "caidas/swin2SR-classical-sr-x2-64",
    "classical-sr-x4": "caidas/swin2SR-classical-sr-x4-64",
    "realworld-sr-x4": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "compressed-sr-x4": "caidas/swin2SR-compressed-sr-x4-48",
    "lightweight-x2": "caidas/swin2SR-lightweight-x2-64",
}


def resolve_swin2sr_model_id(model_name, method):
    if model_name in (None, "default"):
        if method in ("scale4x", "noise_scale4x"):
            return SWIN2SR_MODELS["classical-sr-x4"]
        else:
            return SWIN2SR_MODELS["classical-sr-x2"]
    return SWIN2SR_MODELS.get(model_name, model_name)


class Swin2SRWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.i2i_scale = getattr(model.config, "upscale", 2)
        self.i2i_offset = 0
        self.i2i_blend_size = 8
        self.i2i_default_tile_size = 256
        self.i2i_default_batch_size = 1

    def find_valid_tile_size(self, base_tile_size):
        if base_tile_size is None:
            base_tile_size = self.i2i_default_tile_size
        return (base_tile_size // 8) * 8

    def forward(self, x):
        return self.model(x).reconstruction


class Swin2SRBenchmark:
    def __init__(self, model_name, gpus):
        self.model_name = model_name
        self.device = create_device(gpus)
        self.gpus = gpus
        self.model = None
        self.wrapper = None
        self.is_half = False

    def load_model(self, method, noise_level):
        from nunif.transforms.tta import tta_merge, tta_split

        self.tta_split = tta_split
        self.tta_merge = tta_merge

        model_id = resolve_swin2sr_model_id(self.model_name, method)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
        self.wrapper = Swin2SRWrapper(self.model)
        self._setup()

    def _setup(self):
        if self.wrapper is not None:
            self.wrapper = self.wrapper.to(self.device).eval()

    def half(self):
        self.is_half = True
        if self.wrapper is not None:
            self.wrapper = self.wrapper.half()
        return self

    def float(self):
        self.is_half = False
        if self.wrapper is not None:
            self.wrapper = self.wrapper.float()
        return self

    def compile(self):
        from nunif.models import compile_model

        from .utils import can_compile

        if self.wrapper is not None and can_compile(self.wrapper):
            self.wrapper = compile_model(self.wrapper)

    def warmup(self, tile_size, batch_size, enable_amp):
        if self.wrapper is None:
            return
        valid_tile_size = self.wrapper.find_valid_tile_size(tile_size)
        bs = batch_size or 1
        x = torch.zeros(
            (bs, 3, valid_tile_size, valid_tile_size),
            device=self.device,
            dtype=torch.float16 if self.is_half else torch.float32,
        )
        with autocast(device=self.device, enabled=enable_amp):
            self.wrapper(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def render(self, x, tile_size=None, batch_size=None, enable_amp=False):
        return tiled_render(x, self.wrapper, tile_size=tile_size, batch_size=batch_size, enable_amp=enable_amp)

    def convert(
        self,
        x,
        alpha,
        method,
        noise_level,
        tile_size=None,
        batch_size=None,
        tta=False,
        enable_amp=False,
        output_device="cpu",
    ):
        assert not torch.is_grad_enabled()
        assert x.shape[0] == 3

        x = x.to(self.device)
        if tta:
            rgb = self.tta_merge(
                [
                    self.render(xx, tile_size=tile_size, batch_size=batch_size, enable_amp=enable_amp)
                    for xx in self.tta_split(x)
                ]
            )
        else:
            rgb = self.render(x, tile_size=tile_size, batch_size=batch_size, enable_amp=enable_amp)

        rgb = rgb.to(output_device)
        return rgb, None


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=str, help="model dir")
    if HAVE_TRANSFORMERS:
        parser.add_argument(
            "--swin2sr",
            type=str,
            nargs="?",
            const="default",
            help="Swin2SR model name or directory (caidas/swin2SR-classical-sr-x2-64, etc.)",
        )
    parser.add_argument("--noise-level", "-n", type=int, default=0, choices=[0, 1, 2, 3], help="noise level")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["scale", "scale4x", "noise", "noise_scale", "noise_scale4x"],
        default="scale",
        help="method",
    )
    parser.add_argument(
        "--model-method",
        type=str,
        choices=["scale", "scale4x", "noise", "noise_scale", "noise_scale4x"],
        help="method for target model",
    )
    parser.add_argument("--color", type=str, choices=["rgb", "y", "y_matlab"], default="y_matlab", help="colorspace")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="jpeg quality for noise/noise_scale")
    parser.add_argument("--jpeg-times", type=int, default=1, help="number of repetitions of jpeg compression")
    parser.add_argument(
        "--jpeg-quality-down", type=int, default=5, help="value of jpeg quality that decreases every times"
    )
    parser.add_argument("--jpeg-shift", type=int, default=0, help="shift image block each jpeg compression")
    parser.add_argument("--jpeg-yuv420", action="store_true", help="use yuv420 jpeg")
    parser.add_argument(
        "--filter",
        type=str,
        choices=[
            "catrom",
            "box",
            "lanczos",
            "sinc",
            "triangle",
            "torch_bilinear",
            "torch_bilinear_antialias",
            "torch_bilinear_align_corners",
        ],
        default="catrom",
        help="downscaling filter for generate LR image",
    )
    parser.add_argument("--blur", type=float, default=1, help="resize blur. 0.95: shapen, 1.05: blur")
    parser.add_argument("--baseline", action="store_true", help="show the score of --baseline-filter")
    parser.add_argument(
        "--baseline-filter",
        type=str,
        default="catrom",
        choices=["catrom", "box", "lanczos", "sinc", "triangle"],
        help="baseline filter",
    )
    parser.add_argument("--border", type=int, default=0, help="border px removed from the result image")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, help="output file or directory")
    parser.add_argument("--input", "-i", type=str, required=True, help="input directory. (*.txt, *.csv) for image list")
    parser.add_argument("--tta", action="store_true", help="TTA mode (aka geometric self-ensemble)")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--half", action="store_true", help="Use float16 model. AMP will be disabled.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if possible")
    parser.add_argument(
        "--flat-only", action="store_true", help="Only evaluate flat color area (for color change caused by GAN)"
    )
    args = parser.parse_args()
    logger.debug(vars(args))

    if args.model_dir in ("swin2sr", "swinir"):
        if not HAVE_TRANSFORMERS:
            parser.error("Swin2SR requires 'transformers' which is not installed.")
        args.swin2sr = "default"
        args.model_dir = None

    has_swin2sr = HAVE_TRANSFORMERS and getattr(args, "swin2sr", None) is not None
    if args.model_dir is None and not has_swin2sr:
        parser.error("one of --model-dir or --swin2sr is required")
    if args.model_dir is not None and has_swin2sr:
        parser.error("--model-dir and --swin2sr cannot be used together")

    return args


def main():
    from .utils import Waifu2x

    args = parse_args()
    model_method = args.model_method if args.model_method is not None else args.method
    if getattr(args, "swin2sr", None) is not None:
        ctx = Swin2SRBenchmark(model_name=args.swin2sr, gpus=args.gpu)
        model_name = resolve_swin2sr_model_id(args.swin2sr, model_method)
    else:
        ctx = Waifu2x(model_dir=args.model_dir, gpus=args.gpu)
        model_name = args.model_dir

    if args.flat_only and args.color != "rgb":
        warnings.warn("Use --color rgb for --flat-only")
        args.color = "rgb"

    ctx.load_model(model_method, args.noise_level)
    if args.half:
        ctx.half()
        args.disable_amp = True
    if args.compile:
        args.batch_size = 1
        ctx.compile()
        ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size, enable_amp=not args.disable_amp)

    if path.isdir(args.input):
        files = ImageLoader.listdir(args.input)
    elif path.splitext(args.input)[-1] in (".txt", ".csv"):
        files = load_files(args.input)
    else:
        raise ValueError("Unknown input format")

    loader = ImageLoader(files=files, max_queue_size=128, load_func_kwargs={"color": "rgb"})
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
    with torch.inference_mode():
        mse_sum = psnr_sum = time_sum = 0
        baseline_mse_sum = baseline_psnr_sum = baseline_time_sum = 0
        count = 0

        for im, meta in tqdm(loader, ncols=60):
            x = TF.to_tensor(im)
            groundtruth = NF.crop_mod(x, 4)
            x, groundtruth = make_input_waifu2x(groundtruth, args)
            t = time.time()
            if args.half:
                x = x.half().to(ctx.device)
            z, _ = ctx.convert(
                x,
                None,
                model_method,
                args.noise_level,
                args.tile_size,
                args.batch_size,
                tta=args.tta,
                enable_amp=not args.disable_amp,
            )
            if z.dtype == torch.float16:
                z = z.float()

            time_sum += time.time() - t
            if args.border > 0:
                psnr, mse, ignore = psnr256(
                    remove_border(groundtruth, args.border), remove_border(z, args.border), args.color, args.flat_only
                )
            else:
                psnr, mse, ignore = psnr256(groundtruth, z, args.color, args.flat_only)
            if not ignore:
                psnr_sum += psnr
                mse_sum += mse
                count += 1

            if not ignore and args.baseline:
                t = time.time()
                if args.method in ("scale", "noise_scale"):
                    z = IM.scale(x, 2, filter_type=args.baseline_filter, blur=args.blur)
                elif args.method in ("scale4x", "noise_scale4x"):
                    z = IM.scale(x, 4, filter_type=args.baseline_filter, blur=args.blur)
                else:
                    z = x
                baseline_time_sum += time.time() - t
                if args.border > 0:
                    psnr, mse, _ = psnr256(
                        remove_border(groundtruth, args.border),
                        remove_border(z, args.border),
                        args.color,
                        args.flat_only,
                    )
                else:
                    psnr, mse, _ = psnr256(groundtruth, z, args.color, args.flat_only)
                baseline_psnr_sum += psnr
                baseline_mse_sum += mse

        mpsnr = round(psnr_sum / count, 4)
        rmse = round(math.sqrt(mse_sum / count), 4)
        fps = round(count / time_sum, 4)
        print(f"* {model_name}")
        print(f"PSNR: {mpsnr}, RMSE: {rmse}, time: {round(time_sum, 4)} ({fps} FPS), images: {count}")
        if args.baseline:
            mpsnr = round(baseline_psnr_sum / count, 4)
            rmse = round(math.sqrt(baseline_mse_sum / count), 4)
            fps = round(count / baseline_time_sum, 4)
            if args.method in {"scale", "scale4x"}:
                print(f"* {args.baseline_filter}")
            elif args.method in {"noise_scale", "noise_scale4x"}:
                print(f"* {args.baseline_filter}, jpeg")
            elif args.method == "noise":
                print("* jpeg")
            print(f"PSNR: {mpsnr}, RMSE: {rmse}, time: {round(baseline_time_sum, 4)} ({fps} FPS)")
        if device_is_cuda(ctx.device):
            print("GPU Max Memory Allocated", int(torch.cuda.max_memory_allocated(ctx.device) / (1024 * 1024)), "MB")


if __name__ == "__main__":
    main()
