import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == "__main__":
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
