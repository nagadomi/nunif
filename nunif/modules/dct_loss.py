import torch
import torch.nn as nn
import torch.nn.functional as F
from dctorch.functional import dct2
from .color import rgb_to_yrgb
from . permute import window_partition2d
from . charbonnier_loss import charbonnier_loss


def dct_loss(input, target, loss_function=F.l1_loss, clamp=False):
    if clamp:
        nonclip_loss = F.l1_loss(input, target)
        clip_loss = loss_function(dct2(torch.clamp(input, 0, 1)), dct2(torch.clamp(target, 0, 1)))
        return clip_loss + nonclip_loss * 0.001
    else:
        return loss_function(dct2(input), dct2(target))


def window_dct_loss(input, target, window_size=8, loss_function=F.l1_loss, clamp=False):
    if input.shape[2] % window_size != 0:
        rem = window_size - input.shape[2] % window_size
        pad1 = rem // 2
        pad2 = rem - pad1
        input = F.pad(input, (pad1, pad2, pad1, pad2) * 4, mode="reflect")
        target = F.pad(target, (pad1, pad2, pad1, pad2) * 4, mode="reflect")

    input = window_partition2d(input, window_size=window_size)
    target = window_partition2d(target, window_size=window_size)
    B, N, C, H, W = input.shape
    input = input.reshape(B * N, C, H, W).contiguous()
    target = target.reshape(B * N, C, H, W).contiguous()
    return dct_loss(input, target, loss_function=loss_function, clamp=clamp)


def overlap_window_dct_loss(input, target, window_size=8, loss_function=F.l1_loss, clamp=False):
    assert window_size % 2 == 0
    pad = window_size // 2
    if input.shape[2] % window_size != 0:
        assert input.shape[2] == input.shape[3]
        rem = (window_size - input.shape[2] % window_size)
        pad1 = rem // 2
        pad2 = rem - pad1
        input2 = F.pad(input, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad), mode="reflect")
        target2 = F.pad(target, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad), mode="reflect")
        input = F.pad(input, (pad1, pad2, pad1, pad2), mode="reflect")
        target = F.pad(target, (pad1, pad2, pad1, pad2), mode="reflect")
    else:
        input2 = F.pad(input, (pad,) * 4, mode="reflect")
        target2 = F.pad(target, (pad,) * 4, mode="reflect")

    dct1 = window_dct_loss(input, target, window_size=window_size, loss_function=loss_function, clamp=clamp)
    dct2 = window_dct_loss(input2, target2, window_size=window_size, loss_function=loss_function, clamp=clamp)
    return dct1 * 0.5 + dct2 * 0.5


class DCTLoss(nn.Module):
    # BCHW
    def __init__(self, window_size=None, overlap=False, loss_function="l1", clamp=False):
        super().__init__()
        self.clamp = clamp
        self.window_size = window_size
        self.overlap = overlap
        if isinstance(loss_function, str):
            if loss_function == "l1":
                self.loss_function = F.l1_loss
            elif loss_function in {"l2", "mse"}:
                self.loss_function = F.mse_loss
            elif loss_function == "charbonnier":
                self.loss_function = charbonnier_loss
            else:
                raise ValueError(loss_function)
        else:
            self.loss_function = loss_function

    def forward(self, input, target):
        input = rgb_to_yrgb(input)
        target = rgb_to_yrgb(target)
        if self.window_size is not None:
            if self.overlap:
                return overlap_window_dct_loss(input, target, window_size=self.window_size,
                                               loss_function=self.loss_function, clamp=self.clamp)
            else:
                return window_dct_loss(input, target, window_size=self.window_size,
                                       loss_function=self.loss_function, clamp=self.clamp)
        else:
            return dct_loss(input, target, loss_function=self.loss_function, clamp=self.clamp)


def _test():
    dct = DCTLoss()
    x = torch.rand((4, 3, 32, 32))
    y = x + torch.rand((4, 3, 32, 32)) * 0.01
    print(dct(x, y))

    dct8 = DCTLoss(window_size=8, overlap=False)
    print(dct8(x, y))

    x = torch.rand((4, 3, 30, 30))
    y = x + torch.rand((4, 3, 30, 30)) * 0.01
    dct8 = DCTLoss(window_size=8, overlap=True)
    print(dct8(x, y))


if __name__ == "__main__":
    _test()
