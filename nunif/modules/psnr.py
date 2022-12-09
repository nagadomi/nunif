import torch
from torch import nn


"""
PSNR module designed for evaluation.
Note that this is not a loss function for training.
"""


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        mse = self.mse(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
        return -10 * torch.log10(1.0 / (mse + 1.0e-6))


class LuminancePSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    @staticmethod
    def to_luminance(rgb):
        assert (rgb.shape[1] == 3)
        w = [0.29891, 0.58661, 0.11448]
        return (rgb[:, 0:1, :, :] * w[0] +
                rgb[:, 1:2, :, :] * w[1] +
                rgb[:, 2:3, :, :] * w[2])

    def forward(self, input, target):
        mse = self.mse(torch.clamp(self.to_luminance(input), 0, 1),
                       torch.clamp(self.to_luminance(target), 0, 1))
        psnr = 10 * torch.log10(1.0 / (mse + 1.0e-6))
        return -psnr
