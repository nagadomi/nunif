import torch
import torch.nn as nn
import torch.nn.functional as F
from . multiscale_loss import MultiscaleLoss


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


class FFTLoss(nn.Module):
    # BCHW or CHW
    def __init__(self, norm="backward"):
        super().__init__()
        self.norm = norm

    def forward(self, input, target):
        return fft_loss(input, target, norm=self.norm)


class L1FFTLoss(nn.Module):
    def __init__(self, weight=0.1, norm="backward"):
        super().__init__()
        self.weight = weight
        self.norm = norm

    def forward(self, input, target):
        return F.l1_loss(input, target) + fft_loss(input, target, norm=self.norm) * self.weight


class MultiscaleL1FFTLoss(nn.Module):
    def __init__(self, scale_factors=(1, 2), weights=(0.5, 0.5),
                 mode="bilinear",
                 fft_weight=0.1, norm="backward"):
        super().__init__()
        self.loss = MultiscaleLoss(
            L1FFTLoss(weight=fft_weight, norm=norm),
            scale_factors=scale_factors, weights=weights, mode=mode)

    def forward(self, input, target):
        return self.loss(input, target)


def _test():
    criterion = L1FFTLoss(norm="ortho").cuda()

    x = torch.rand((1, 3, 32, 32)).cuda()
    t = torch.rand((1, 3, 32, 32)).cuda()

    with torch.autocast(device_type="cuda"):
        print(criterion(x, t))


if __name__ == "__main__":
    _test()
