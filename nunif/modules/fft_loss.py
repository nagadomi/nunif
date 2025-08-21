import torch
import torch.nn as nn
import torch.nn.functional as F
from . multiscale_loss import MultiscaleLoss
from . clamp_loss import ClampLoss
from . weighted_loss import WeightedLoss
from . gradient_loss import GradientLoss
from . color import RGBToYRGB, rgb_to_yrgb
from . permute import window_partition2d

# ref. Rethinking Coarse-to-Fine Approach in Single Image Deblurring
# https://arxiv.org/abs/2108.05054


fp16_max_value = torch.finfo(torch.float16).max - 1


def ste_clamp(x, overshoot_scale=0.01):
    x_clamp = x.clamp(0, 1)
    x = x_clamp + (x - x_clamp) * overshoot_scale
    return x + x.clamp(0, 1).detach() - x.detach()


# NOTE: Lately I have realized that norm should be `ortho`.
#       otherwise, the gradient norm depends on the resolution of the input image.
def fft_loss(input, target, norm="backward", padding=0, use_phase=True):
    if padding > 0:
        input = F.pad(input, [padding] * 4)
        target = F.pad(target, [padding] * 4)

    if input.dtype == torch.float16:
        input = torch.fft.fft2(input.to(torch.float32), norm=norm, dim=(-2, -1))
        target = torch.fft.fft2(target.to(torch.float32), norm=norm, dim=(-2, -1))
        if use_phase:
            input = torch.stack([input.real, input.imag], dim=-1)
            target = torch.stack([target.real, target.imag], dim=-1)
        else:
            input = torch.abs(input)
            target = torch.abs(target)

        loss = torch.clamp(F.l1_loss(input, target), -fp16_max_value, fp16_max_value).to(torch.float16)
    else:
        input = torch.fft.fft2(input, norm=norm, dim=(-2, -1))
        target = torch.fft.fft2(target, norm=norm, dim=(-2, -1))
        if use_phase:
            input = torch.stack([input.real, input.imag], dim=-1)
            target = torch.stack([target.real, target.imag], dim=-1)
        else:
            input = torch.abs(input)
            target = torch.abs(target)

        loss = F.l1_loss(input, target)
    return loss


def window_fft_loss(input, target, window_size=8, norm="backward", padding=0, use_phase=True):
    input = window_partition2d(input, window_size=window_size)
    target = window_partition2d(target, window_size=window_size)
    B, N, C, H, W = input.shape
    input = input.reshape(B * N, C, H, W).contiguous()
    target = target.reshape(B * N, C, H, W).contiguous()
    return fft_loss(input, target, norm=norm, padding=padding, use_phase=use_phase)


def overlap_window_fft_loss(input, target, window_size=8, norm="backward", padding=0, use_phase=True):
    assert window_size % 2 == 0
    pad = window_size // 2
    if input.shape[2] % window_size != 0:
        assert input.shape[2] == input.shape[3]
        rem = (window_size - input.shape[2] % window_size)
        pad1 = rem // 2
        pad2 = rem - pad1
        input2 = F.pad(input, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad))
        target2 = F.pad(target, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad))
        input = F.pad(input, (pad1, pad2, pad1, pad2))
        target = F.pad(target, (pad1, pad2, pad1, pad2))
    else:
        input2 = F.pad(input, (pad,) * 4)
        target2 = F.pad(target, (pad,) * 4)

    dct1 = window_fft_loss(input, target, window_size=window_size, norm=norm, padding=padding, use_phase=use_phase)
    dct2 = window_fft_loss(input2, target2, window_size=window_size, norm=norm, padding=padding, use_phase=use_phase)
    return (dct1 + dct2) * 0.5


class FFTLoss(nn.Module):
    # BCHW or CHW
    def __init__(self, norm="backward", window_size=None, padding=0, use_phase=True, overlap=False):
        super().__init__()
        self.norm = norm
        self.overlap = overlap
        self.use_phase = use_phase
        self.window_size = window_size
        self.padding = padding

    def forward(self, input, target):
        if self.window_size is not None:
            if self.overlap:
                return overlap_window_fft_loss(input, target, window_size=self.window_size,
                                               norm=self.norm, padding=self.padding, use_phase=self.use_phase)
            else:
                return window_fft_loss(input, target, window_size=self.window_size,
                                       norm=self.norm, padding=self.padding, use_phase=self.use_phase)
        else:
            return fft_loss(input, target, norm=self.norm, padding=self.padding, use_phase=self.use_phase)


def L1FFTLoss(weight=0.1, norm="backward", window_size=None, padding=0):
    return WeightedLoss((nn.L1Loss(), FFTLoss(norm=norm, window_size=window_size, padding=padding)), weights=(1.0, weight))


def YRGBL1FFTLoss(weight=0.1, norm="backward", window_size=None, padding=0):
    return WeightedLoss((ClampLoss(nn.L1Loss()), FFTLoss(norm=norm, window_size=window_size, padding=padding)),
                        weights=(1.0, weight), preprocess=RGBToYRGB())


def YRGBL1FFTGradientLoss(fft_weight=0.1, grad_weight=0.1, norm="backward", diag=False, window_size=None, padding=0):
    return WeightedLoss((ClampLoss(nn.L1Loss()),
                         FFTLoss(norm=norm, window_size=window_size, padding=padding),
                         ClampLoss(GradientLoss(diag=diag))),
                        weights=(1.0, fft_weight, grad_weight), preprocess=RGBToYRGB())


class MultiscaleL1FFTLoss(nn.Module):
    def __init__(self, scale_factors=(1, 2), weights=(0.5, 0.5),
                 mode="bilinear",
                 fft_weight=0.1, norm="backward", window_size=None, padding=0):
        super().__init__()
        self.loss = MultiscaleLoss(
            L1FFTLoss(weight=fft_weight, norm=norm, window_size=window_size, padding=padding),
            scale_factors=scale_factors, weights=weights, mode=mode)

    def forward(self, input, target):
        return self.loss(input, target)


def _test():
    criterion = L1FFTLoss(norm="ortho").cuda()

    x = torch.rand((1, 3, 32, 32)).cuda()
    t = torch.rand((1, 3, 32, 32)).cuda()

    with torch.autocast(device_type="cuda"):
        print(criterion(x, t))

    criterion = FFTLoss(window_size=8, padding=8).cuda()
    print(criterion(x, t))


def _test_grad():
    import torchvision.io as io
    from .lbp_loss import YRGBLBP

    y = io.read_image("cc0/320/dog.png") / 255.0
    y = y.unsqueeze(0)
    x = y + (torch.rand_like(y) * 0.5)
    x.requires_grad_(True)

    x.grad = None
    l1_loss = nn.L1Loss()
    l1_loss(x, y).backward()
    l1_norm, l1_max = x.grad.norm(), x.grad.abs().max()

    x.grad = None
    fft_loss = FFTLoss(norm="ortho", window_size=32, use_phase=False)
    (fft_loss(x, y.detach())).backward()
    fft_norm, fft_max = x.grad.norm(), x.grad.abs().max()

    print("fft loss norm", fft_norm, "max", fft_max)
    print("l1 loss norm", l1_norm, "max", l1_max, "weight", l1_norm / fft_norm)


if __name__ == "__main__":
    _test()
    _test_grad()
