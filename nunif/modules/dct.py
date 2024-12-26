import scipy.fft
import numpy as np
import torch
import torch.nn as nn

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


if __name__ == "__main__":
    _test()
