import torch
from torch import nn


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        mse = self.mse(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
        return -10 * torch.log10(1.0 / (mse + 1.0e-6))
