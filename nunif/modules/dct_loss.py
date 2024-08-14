import torch
import torch.nn as nn
import torch.nn.functional as F
from dctorch.functional import dct2
from .color import rgb_to_yrgb
from . permute import window_partition2d
from . charbonnier_loss import charbonnier_loss
from . transforms import diff_rotate


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
        input = F.pad(input, (pad1, pad2, pad1, pad2))
        target = F.pad(target, (pad1, pad2, pad1, pad2))

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
        input2 = F.pad(input, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad))
        target2 = F.pad(target, (pad1 + pad, pad2 + pad, pad1 + pad, pad2 + pad))
        input = F.pad(input, (pad1, pad2, pad1, pad2))
        target = F.pad(target, (pad1, pad2, pad1, pad2))
    else:
        input2 = F.pad(input, (pad,) * 4)
        target2 = F.pad(target, (pad,) * 4)

    dct1 = window_dct_loss(input, target, window_size=window_size, loss_function=loss_function, clamp=clamp)
    dct2 = window_dct_loss(input2, target2, window_size=window_size, loss_function=loss_function, clamp=clamp)
    return (dct1 + dct2) * 0.5


class DCTLoss(nn.Module):
    # BCHW
    def __init__(self, window_size=None, overlap=False, loss_function="l1", clamp=False, diag=False, random_rotate=False):
        super().__init__()
        self.clamp = clamp
        self.window_size = window_size
        self.overlap = overlap
        self.diag = diag
        self.random_rotate = random_rotate

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

    def forward_loss(self, input, target):
        if self.window_size is not None:
            if self.overlap:
                return overlap_window_dct_loss(input, target, window_size=self.window_size,
                                               loss_function=self.loss_function, clamp=self.clamp)
            else:
                return window_dct_loss(input, target, window_size=self.window_size,
                                       loss_function=self.loss_function, clamp=self.clamp)
        else:
            return dct_loss(input, target, loss_function=self.loss_function, clamp=self.clamp)

    def forward(self, input, target):
        input = rgb_to_yrgb(input)
        target = rgb_to_yrgb(target)
        loss1 = self.forward_loss(input, target)

        if self.random_rotate:
            if self.training:
                angle = torch.rand(1).item() * 360
            else:
                angle = 45
            loss2 = self.forward_loss(diff_rotate(input, angle, expand=True, padding_mode="zeros"),
                                      diff_rotate(target, angle, expand=True, padding_mode="zeros"))
            return (loss1 + loss2) * 0.5
        elif self.diag:
            loss2 = self.forward_loss(diff_rotate(input, 45, expand=True, padding_mode="zeros"),
                                      diff_rotate(target, 45, expand=True, padding_mode="zeros"))
            return (loss1 + loss2) * 0.5
        else:
            return loss1


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
