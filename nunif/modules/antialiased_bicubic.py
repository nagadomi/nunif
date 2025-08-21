# Antialised Bicubic for ONNX
# This is just downscaling after a Gaussian blur without antialiasing,
# so it does not exactly match pytorch bicubic with antialiasing.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gaussian_filter import GaussianFilter2d


class AntialiasedBicubic(nn.Module):
    def __init__(self, in_channels, downscale_factor, sigma=None):
        super().__init__()
        assert downscale_factor in {2, 4}
        self.downscale_factor = downscale_factor
        if sigma is None:
            sigma = {2: 0.74, 4: 0.66}[self.downscale_factor]
        self.antialias = GaussianFilter2d(in_channels, kernel_size=3, sigma=sigma, padding=1)

    def forward(self, x):
        if self.downscale_factor == 2:
            x = self.antialias(x)
            x = F.interpolate(x, scale_factor=0.5,
                              mode="bicubic", antialias=False, align_corners=False)
        else:
            x = self.antialias(x)
            x = F.interpolate(x, scale_factor=0.5,
                              mode="bicubic", antialias=False, align_corners=False)
            x = self.antialias(x)
            x = F.interpolate(x, scale_factor=0.5,
                              mode="bicubic", antialias=False, align_corners=False)
        return x


def _find_sigma():
    import torchvision.io as io

    xs = (
        (io.read_image("cc0/320/bottle.png") / 256).unsqueeze(0),
        (io.read_image("cc0/320/dog.png") / 256).unsqueeze(0),
        (io.read_image("cc0/320/light_house.png") / 256).unsqueeze(0)
    )

    for downscale_factor in [2, 4]:
        z2s = [F.interpolate(x, scale_factor=1 / downscale_factor, mode="bicubic", antialias=True, align_corners=False)
               for x in xs]
        min_diff = 1000
        min_sigma = None
        for sigma in torch.arange(0.1, 2.5, 0.01):
            resize = AntialiasedBicubic(3, downscale_factor=downscale_factor, sigma=sigma)
            z1s = [resize(x) for x in xs]

            diff = sum([(z1 - z2).abs().mean() for z1, z2 in zip(z1s, z2s)])
            if diff < min_diff:
                min_diff = diff
                min_sigma = sigma

        print(downscale_factor, "sigma", min_sigma, "mean", min_diff)


if __name__ == "__main__":
    _find_sigma()
