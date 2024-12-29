import scipy.fft
import numpy as np
import torch
import torch.nn as nn
from .permute import window_partition2d
from .init import basic_module_init


# Exportable Compilable DCT2 module
# Most codes adapted from dctorch
# https://github.com/GallagherCommaJack/dctorch


class DCT2(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            h, w = size, size
        else:
            h, w = size

        self.register_buffer("mh", torch.tensor(scipy.fft.dct(np.eye(h), norm="ortho"), dtype=torch.float32))
        self.register_buffer("mw", torch.tensor(scipy.fft.dct(np.eye(w), norm="ortho"), dtype=torch.float32))
        self.h = h
        self.w = w

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h == self.h and w == self.w
        return torch.einsum("...hw,hi,wj->...ij", x, self.mh, self.mw)


class IDCT2(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            h, w = size, size
        else:
            h, w = size

        self.register_buffer("mh", torch.tensor(scipy.fft.idct(np.eye(h), norm="ortho"), dtype=torch.float32))
        self.register_buffer("mw", torch.tensor(scipy.fft.idct(np.eye(w), norm="ortho"), dtype=torch.float32))
        self.h = h
        self.w = w

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h == self.h and w == self.w
        return torch.einsum("...hw,hi,wj->...ij", x, self.mh, self.mw)


class WindowDCT(nn.Module):
    # non overlap window dct
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.dct2 = DCT2(window_size)

    def window_dct(self, x):
        B, C, H, W = x.shape
        x = window_partition2d(x, self.window_size)
        x = self.dct2(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, C * self.window_size ** 2, H // self.window_size, W // self.window_size)
        return x.contiguous()

    def forward(self, x):
        return self.window_dct(x)


class ChannelIDCT(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, project=True):
        super().__init__()
        self.out_channels = out_channels
        self.window_size = window_size
        if project:
            self.proj = nn.Conv2d(in_channels, out_channels * window_size ** 2, kernel_size=1, stride=1, padding=0)
        else:
            self.proj = nn.Identity()
            assert in_channels == out_channels * window_size ** 2

        if self.window_size > 1:
            self.idct2 = IDCT2(self.window_size)
        else:
            self.idct2 = None
        basic_module_init(self.proj)

    def channel_idct(self, x):
        B, C, H, W = x.shape
        OC = self.out_channels
        window_size = int((C / OC) ** 0.5)
        assert C % OC == 0 and (C // OC) ** 0.5 == window_size

        x = x.permute(0, 2, 3, 1).reshape(B, H, W, OC, window_size, window_size)
        x = self.idct2(x)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, OC, H * window_size, W * window_size)
        return x

    def forward(self, x):
        x = self.proj(x)
        if self.window_size > 1:
            x = self.channel_idct(x)
        return x


def _test():
    from dctorch.functional import dct2, idct2
    import torchvision.io as IO
    import math

    x = (IO.read_image("cc0/320/dog.png") / 256.0).unsqueeze(0)
    my_dct2 = DCT2((x.shape[2], x.shape[3]))
    my_idct2 = IDCT2((x.shape[2], x.shape[3]))

    ref_dct_ret = dct2(x)
    my_dct_ret = my_dct2(x)
    assert math.isclose((ref_dct_ret - my_dct_ret).abs().sum(), 0)

    ref_idct_ret = idct2(ref_dct_ret)
    my_idct_ret = my_idct2(my_dct_ret)
    assert math.isclose((ref_idct_ret - my_idct_ret).abs().sum(), 0)


def _test_window_dct():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import math

    src = (IO.read_image("cc0/320/dog.png") / 256.0)
    dct = WindowDCT(window_size=8)
    idct = ChannelIDCT(3 * 8 ** 2, 3, window_size=8, project=False)

    x = dct(src.unsqueeze(0))
    x = idct(x).squeeze(0)

    TF.to_pil_image(x).show()

    assert math.isclose((x - src).abs().mean(), 0, abs_tol=1e-6)


if __name__ == "__main__":
    _test()
    _test_window_dct()
