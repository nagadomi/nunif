import torch
import torch.nn as nn
import torch.nn.functional as F
from . multiscale_loss import MultiscaleLoss
from . clamp_loss import ClampLoss
from . weighted_loss import WeightedLoss
from . gradient_loss import GradientLoss
from . color import RGBToYRGB, rgb_to_yrgb
from . lbp_loss import YLBP
from . permute import window_partition2d

# ref. Rethinking Coarse-to-Fine Approach in Single Image Deblurring
# https://arxiv.org/abs/2108.05054


fp16_max_value = torch.finfo(torch.float16).max - 1


def fft_loss(input, target, norm="backward"):
    if input.dtype == torch.float16:
        input = torch.fft.fft2(input.to(torch.float32), norm=norm, dim=(-2, -1))
        target = torch.fft.fft2(target.to(torch.float32), norm=norm, dim=(-2, -1))
        input = torch.stack([input.real, input.imag], dim=-1)
        target = torch.stack([target.real, target.imag], dim=-1)
        loss = torch.clamp(F.l1_loss(input, target), -fp16_max_value, fp16_max_value).to(torch.float16)
    else:
        input = torch.fft.fft2(input, norm=norm, dim=(-2, -1))
        target = torch.fft.fft2(target, norm=norm, dim=(-2, -1))
        input = torch.stack([input.real, input.imag], dim=-1)
        target = torch.stack([target.real, target.imag], dim=-1)
        loss = F.l1_loss(input, target)
    return loss


def window_fft_loss(input, target, window_size=8, norm="backward"):
    input = window_partition2d(input, window_size=window_size)
    target = window_partition2d(target, window_size=window_size)
    B, N, C, H, W = input.shape
    input = input.reshape(B * N, C, H, W).contiguous()
    target = target.reshape(B * N, C, H, W).contiguous()
    return fft_loss(input, target, norm=norm)


class FFTLoss(nn.Module):
    # BCHW or CHW
    def __init__(self, norm="backward", window_size=None):
        super().__init__()
        self.norm = norm
        self.window_size = window_size

    def forward(self, input, target):
        if self.window_size is not None:
            return window_fft_loss(input, target, window_size=self.window_size, norm=self.norm)
        else:
            return fft_loss(input, target, norm=self.norm)


class LBPFFTLoss(nn.Module):
    def __init__(self, kernel_size=3, weight=0.1, norm="backward", window_size=None):
        super().__init__()
        self.norm = norm
        self.lbp = YLBP(kernel_size=kernel_size)
        self.weight = weight
        self.window_size = window_size

    def forward(self, input, target):
        lbp = self.lbp(input, target)
        if self.window_size is not None:
            fft = window_fft_loss(rgb_to_yrgb(input), rgb_to_yrgb(target),
                                  window_size=self.window_size, norm=self.norm)
        else:
            fft = fft_loss(rgb_to_yrgb(input), rgb_to_yrgb(target), norm=self.norm)
        return lbp + fft * self.weight


def L1FFTLoss(weight=0.1, norm="backward", window_size=None):
    return WeightedLoss((nn.L1Loss(), FFTLoss(norm=norm, window_size=window_size)), weights=(1.0, weight))


def YRGBL1FFTLoss(weight=0.1, norm="backward", window_size=None):
    return WeightedLoss((ClampLoss(nn.L1Loss()), FFTLoss(norm=norm, window_size=window_size)),
                        weights=(1.0, weight), preprocess=RGBToYRGB())


def YRGBL1FFTGradientLoss(fft_weight=0.1, grad_weight=0.1, norm="backward", diag=False, window_size=None):
    return WeightedLoss((ClampLoss(nn.L1Loss()),
                         FFTLoss(norm=norm, window_size=window_size),
                         ClampLoss(GradientLoss(diag=diag))),
                        weights=(1.0, fft_weight, grad_weight), preprocess=RGBToYRGB())


class MultiscaleL1FFTLoss(nn.Module):
    def __init__(self, scale_factors=(1, 2), weights=(0.5, 0.5),
                 mode="bilinear",
                 fft_weight=0.1, norm="backward", window_size=None):
        super().__init__()
        self.loss = MultiscaleLoss(
            L1FFTLoss(weight=fft_weight, norm=norm, window_size=window_size),
            scale_factors=scale_factors, weights=weights, mode=mode)

    def forward(self, input, target):
        return self.loss(input, target)


def _test():
    criterion = L1FFTLoss(norm="ortho").cuda()

    x = torch.rand((1, 3, 32, 32)).cuda()
    t = torch.rand((1, 3, 32, 32)).cuda()

    with torch.autocast(device_type="cuda"):
        print(criterion(x, t))

    criterion = FFTLoss(window_size=8).cuda()
    print(criterion(x, t))


if __name__ == "__main__":
    _test()
