import argparse
import os
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as io
import torchvision.transforms.functional as TF
from nunif.modules.dinov2 import DINOv2Loss, DINOv2PoolLoss
from nunif.modules.lpips import LPIPSWith
from dino.models.l4sn import L4SNLoss
from dctorch.functional import dct2
import random


try:
    from nunif.modules.dists import DISTS
    from nunif.modules.fdl_loss import FDLLoss
except ImportError:
    pass


def init_jpeg(x):
    x = x.mul(255).round_().to(torch.uint8)
    jpeg = io.encode_jpeg(x, quality=10)
    x = io.decode_jpeg(jpeg)
    return x / 255.0


def init_noise(x):
    x = torch.rand_like(x).clamp(0.2, 0.8)
    return x


def save_image(x, fn):
    TF.to_pil_image(x.clamp(0, 1)).save(fn)


def ste_clamp(x, overshoot_scale=0.1):
    x_clamp = x.clamp(0, 1)
    x = x_clamp + (x - x_clamp) * overshoot_scale
    return x + x.clamp(0, 1).detach() - x.detach()


class NullLoss(nn.Module):
    def forward(self, x, y):
        return torch.zeros((1,), dtype=x.dtype, device=x.device).mean()


class SlicedWasserstein(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x, y are already random projected
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        x, _ = torch.sort(x, dim=-1)
        y, _ = torch.sort(y, dim=-1)

        return F.l1_loss(x, y)


class PatchSlicedWasserstein(nn.Module):
    def __init__(self, window_size=4):
        super().__init__()
        self.window_size = 4
        self.register_buffer("random_projection", torch.randn((384, 384, 5, 5)) * (384 ** -0.5))

    def forward(self, x, y):
        x = F.conv2d(x, weight=self.random_projection, bias=None)
        y = F.conv2d(y, weight=self.random_projection, bias=None)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        x, _ = torch.sort(x, dim=-1)
        y, _ = torch.sort(y, dim=-1)

        return F.l1_loss(x, y)


class PositionalSlicedWasserstein(nn.Module):
    def __init__(self, window_size=4):
        super().__init__()
        self.window_size = 4
        self.register_buffer("random_projection", torch.randn((384, 384, 1, 1)) * (384 ** -0.5))
        self.pos_cache = None

    def pos_embed(self, x):
        if self.pos_cache is None:
            with torch.no_grad():
                B, C, H, W = x.shape
                y = torch.linspace(-1, 1, steps=H, device=x.device)
                x = torch.linspace(-1, 1, steps=W, device=x.device)
                grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
                pos = torch.stack([grid_y, grid_x], dim=-1).permute(2, 0, 1).unsqueeze(0)
                proj = torch.randn((C, 2, 1, 1), dtype=x.dtype, device=x.device) * (C ** -0.5)
                self.pos_cache = F.conv2d(pos, weight=proj, bias=None)

        return x + self.pos_cache

    def forward(self, x, y):
        # pos embed
        x = self.pos_embed(x)
        y = self.pos_embed(y)
        # random projection
        x = F.conv2d(x, weight=self.random_projection, bias=None)
        y = F.conv2d(y, weight=self.random_projection, bias=None)
        # swd
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        x, _ = torch.sort(x, dim=-1)
        y, _ = torch.sort(y, dim=-1)

        return F.l1_loss(x, y)


class DCTLoss(nn.Module):
    def forward(self, x, y):
        x = dct2(x)
        y = dct2(y)
        return F.l1_loss(x, y)


def shift(x, w, h):
    x = F.pad(x, (w, 0, h, 0), mode="replicate")
    x = F.pad(x, (0, -w, 0, -h), mode="replicate")
    return x


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--init", type=str, choices=["noise", "jpeg", "shift", "resize"], default="noise", help="initial image")
    parser.add_argument("--init-image", type=str, help="initial image")
    parser.add_argument("--model", type=str,
                        choices=["dct", "pool", "swd", "patch-swd", "pos-swd", "dists", "lpips", "fdl",
                                 "l4sn", "l4sn-swd"],
                        required=True)
    parser.add_argument("--iteration", type=int, default=20000, help="iteration")
    parser.add_argument("--fp32", action="store_true", help="use fp32")
    parser.add_argument("--save-interval", type=int, default=100, help="save interval")
    parser.add_argument("--disable-random-shift", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    y = io.read_image(args.input) / 255.
    # y = y[:, :224, :224]

    if args.init_image is not None:
        x = io.read_image(args.init_image) / 255.0
        assert x.shape == y.shape
    else:
        match args.init:
            case "jpeg":
                x = init_jpeg(y)
            case "noise":
                x = init_noise(y)
            case "shift":
                x = shift(y, 3, 3)
            case "resize":
                x = y.unsqueeze(0)
                x = F.interpolate(x, size=(x.shape[2] // 4,x.shape[3] // 4),
                                  mode="bilinear", align_corners=False, antialias=True)
                x = F.interpolate(x, size=(x.shape[2] * 4,x.shape[3] * 4),
                                  mode="bilinear", align_corners=False)
                x = x.squeeze(0)

    x = x.unsqueeze(0).cuda()
    y = y.unsqueeze(0).cuda()
    x.requires_grad_(True)
    y.requires_grad_(False)

    match args.model:
        case "dct":
            model = DINOv2Loss(DCTLoss(), random_projection=64).cuda()
        case "pool":
            model = DINOv2PoolLoss().cuda()
        case "swd":
            model = DINOv2Loss(SlicedWasserstein(), random_projection=384).cuda()
        case "patch-swd":
            model = DINOv2Loss(PatchSlicedWasserstein(), random_projection=None).cuda()
        case "pos-swd":
            model = DINOv2Loss(PositionalSlicedWasserstein(), random_projection=None).cuda()
        case "dists":
            model = DISTS().cuda()
        case "lpips":
            model = LPIPSWith(NullLoss(), 1.0).cuda()
        case "fdl":
            model = FDLLoss().eval().cuda()
        case "l4sn":
            model = L4SNLoss(activation=True)
            model = model.eval().cuda()
        case "l4sn-swd":
            model = L4SNLoss(activation=True, swd_weight=0.5, swd_indexes=[0, 1], swd_window_size=8)
            model = model.eval().cuda()

    optimizer = torch.optim.Adam([x], lr=1e-3, betas=(0.9, 0.99))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=not args.fp32)
    for i in range(args.iteration):
        optimizer.zero_grad()

        if not args.disable_random_shift:
            w = random.randint(-8, 8)
            h = random.randint(-8, 8)
            xx = shift(x, w, h)
            yy = shift(y, w, h)
        else:
            xx = x
            yy = y

        with torch.autocast(device_type="cuda", enabled=not args.fp32):
            loss = model(ste_clamp(xx), yy)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if i % args.save_interval == 0:
            print(i, "loss", loss.item())
            save_image(x[0], path.join(args.output, f"{i}.png"))


if __name__ == "__main__":
    main()
