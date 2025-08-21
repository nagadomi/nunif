import torch.nn as nn
import torch.nn.functional as F
import random


class Pad(nn.Module):
    """
    A wrapper for F.pad.
    Note that F.pad is not specialized for BCHW.
    pad=(1, 1) applies padding only to the last dimension(=W).
    Also supports mode="zeros" for compatibility with Conv2d.
    """
    def __init__(self, pad, mode="zeros", value=0.):
        super().__init__()
        if mode == "zeros":
            value = 0.
            mode = "constant"
        self.mode = mode
        self.value = value
        self.pad = pad

    def forward(self, x):
        return F.pad(x, pad=self.pad, mode=self.mode, value=self.value)


def get_fit_pad_size(x, target, center=True):
    pad_h = target.shape[-2] - x.shape[-2]
    pad_w = target.shape[-1] - x.shape[-1]

    if center:
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
    else:
        pad_w1 = 0
        pad_w2 = pad_w
        pad_h1 = 0
        pad_h2 = pad_h

    return pad_w1, pad_w2, pad_h1, pad_h2


def get_pad_size(x, mod, center=True, random_shift=False):
    if x.shape[-1] % mod == 0:
        pad_w = 0
    else:
        pad_w = (mod - x.shape[-1] % mod)
    if x.shape[-2] % mod == 0:
        pad_h = 0
    else:
        pad_h = (mod - x.shape[-2] % mod)

    if random_shift:
        pad_w1 = random.randint(0, pad_w)
        pad_w2 = pad_w - pad_w1
        pad_h1 = random.randint(0, pad_h)
        pad_h2 = pad_h - pad_h1
    else:
        if center:
            pad_w1 = pad_w // 2
            pad_w2 = pad_w - pad_w1
            pad_h1 = pad_h // 2
            pad_h2 = pad_h - pad_h1
        else:
            pad_w1 = 0
            pad_w2 = pad_w
            pad_h1 = 0
            pad_h2 = pad_h

    return pad_w1, pad_w2, pad_h1, pad_h2


def get_crop_size(x, mod, center=True, random_shift=False):
    if x.shape[-1] % mod == 0:
        pad_w = 0
    else:
        pad_w = x.shape[-1] % mod
    if x.shape[-2] % mod == 0:
        pad_h = 0
    else:
        pad_h = x.shape[-2] % mod

    if random_shift:
        pad_w1 = random.randint(0, pad_w)
        pad_w2 = pad_w - pad_w1
        pad_h1 = random.randint(0, pad_h)
        pad_h2 = pad_h - pad_h1
    else:
        if center:
            pad_w1 = pad_w // 2
            pad_w2 = pad_w - pad_w1
            pad_h1 = pad_h // 2
            pad_h2 = pad_h - pad_h1
        else:
            pad_w1 = 0
            pad_w2 = pad_w
            pad_h1 = 0
            pad_h2 = pad_h

    return -pad_w1, -pad_w2, -pad_h1, -pad_h2


def _test():
    import torch
    x = torch.zeros((1, 3, 4, 4))
    pad1 = (1, 1, 1, 1)
    pad2 = (0, 1, 0, 1)
    pad3 = (-1, -1, -1, -1)
    print(pad1, x.shape, Pad(pad1)(x).shape)
    print(pad2, x.shape, Pad(pad2)(x).shape)
    print(pad3, x.shape, Pad(pad3)(x).shape)

    for mode in ("zeros", "reflect", "replicate", "constant"):
        Pad((1, 1, 1, 1), mode)(x)


def _test_get_pad_size():
    import torch

    x = torch.zeros((4, 3, 34, 33))
    for mod in [2, 3, 4, 16, 18]:
        pad = get_pad_size(x, mod)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0

    for mod in [2, 3, 4, 16, 18]:
        pad = get_pad_size(x, mod, center=False)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0

    for mod in [2, 3, 4, 16, 18]:
        pad = get_pad_size(x, mod, random_shift=True)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0


def _test_get_crop_size():
    import torch

    x = torch.zeros((4, 3, 34, 33))
    for mod in [2, 3, 4, 16, 18]:
        pad = get_crop_size(x, mod)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0

    for mod in [2, 3, 4, 16, 18]:
        pad = get_crop_size(x, mod, center=False)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0

    for mod in [2, 3, 4, 16, 18]:
        pad = get_crop_size(x, mod, random_shift=True)
        x = F.pad(x, pad)
        h, w = x.shape[-2:]
        assert h % mod == 0
        assert w % mod == 0


def _test_get_fit_pad_size():
    import torch

    x = torch.zeros((4, 3, 32, 32))
    for diff in [-2, 2, -5, 5]:
        target = torch.zeros((4, 3, x.shape[-2] + diff, x.shape[-1] + diff))
        pad = get_fit_pad_size(x, target)
        x = F.pad(x, pad)
        x.shape == target.shape


if __name__ == "__main__":
    # _test()
    _test_get_pad_size()
    _test_get_crop_size()
    _test_get_fit_pad_size()
