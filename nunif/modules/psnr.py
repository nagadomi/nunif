import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PSNRPerImage(nn.Module):
    def forward(self, input, target):
        if input.ndim == 4:
            input = torch.clamp(input, 0, 1)
            target = torch.clamp(target, 0, 1)
            psnr_sum = 0
            for x, y in zip(input, target):
                mse = F.mse_loss(x, y)
                psnr = -10 * torch.log10(1.0 / (mse + 1.0e-6))
                psnr_sum = psnr_sum + psnr
            return psnr_sum / input.shape[0]
        else:
            mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
            return -10 * torch.log10(1.0 / (mse + 1.0e-6))


class LuminancePSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    @staticmethod
    def to_luminance(rgb):
        if rgb.shape[1] == 3:
            w = [0.29891, 0.58661, 0.11448]
            return (rgb[:, 0:1, :, :] * w[0] +
                    rgb[:, 1:2, :, :] * w[1] +
                    rgb[:, 2:3, :, :] * w[2])
        else:
            assert rgb.shape[1] == 1  # y
            return rgb

    def forward(self, input, target):
        mse = self.mse(torch.clamp(self.to_luminance(input), 0, 1),
                       torch.clamp(self.to_luminance(target), 0, 1))
        psnr = 10 * torch.log10(1.0 / (mse + 1.0e-6))
        return -psnr
