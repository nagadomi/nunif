import torch


def rgb_to_ycbcr(x, yycbcr=False):
    # Bt.601, -1 .. 1 scale
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]

    y = r * 0.299 + g * 0.587 + b * 0.114
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    if not yycbcr:
        x = torch.cat([y, cb, cr], dim=1)
    else:
        x = torch.cat([y, y, cb, cr], dim=1)
    x = x * 2. - 1.

    return x


class RGBToYCbCr(torch.nn.Module):
    def __init__(self, yycbcr=False):
        super().__init__()
        self.yycbcr = yycbcr

    def forward(self, x):
        return rgb_to_ycbcr(x, self.yycbcr)


def rgb_to_yrgb(x, yycbcr=False):
    y = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
    return torch.cat([y, x], dim=1)


class RGBToYRGB(torch.nn.Module):
    def forward(self, x):
        return rgb_to_yrgb(x)
