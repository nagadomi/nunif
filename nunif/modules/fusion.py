import torch
import torch.nn as nn
import torch.nn.functional as F


class Lerp(nn.Module):
    def __init__(self, weight_shape=None):
        super().__init__()
        if weight_shape is None:
            weight_shape = 1
        if torch.is_tensor(weight_shape):
            self.weight = nn.Parameter(weight_shape.detach().clone())
        else:
            self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=torch.float32))


    def forward(self, input, end):
        # out = input + (0. 5 + self.weight) * (end - start)
        return torch.lerp(input, end, (0.5 + self.weight).to(input.dtype))


class PLerp2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        w = F.sigmoid(self.proj(torch.cat((x1, x2), dim=1)))
        return torch.lerp(x1, x2, w)
