import torch
from torch import nn


class ClampLoss(nn.Module):
    """ Wrapper Module for `(clamp(input, 0, 1) - clamp(target, 0, 1))`
    """
    def __init__(self, module, min_value=0, max_value=1, eta=0.001):
        super().__init__()
        self.module = module
        self.min_value = min_value
        self.max_value = max_value
        self.eta = eta

    def forward(self, input, target):
        noclip_loss = self.module(input, target)
        clip_loss = self.module(torch.clamp(input, self.min_value, self.max_value),
                                torch.clamp(target, self.min_value, self.max_value))

        return clip_loss + noclip_loss * self.eta
