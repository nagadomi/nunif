import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_loss(input, target, loss_function, min_value, max_value, eta=0.001):
    noclip_loss = loss_function(input, target)
    clip_loss = loss_function(torch.clamp(input, min_value, max_value),
                              torch.clamp(target, min_value, max_value))
    return clip_loss + noclip_loss * eta


def clamp_l1_loss(input, target, loss_function, min_value, max_value, eta=0.001):
    noclip_loss = F.l1_loss(input, target)
    clip_loss = loss_function(torch.clamp(input, min_value, max_value),
                              torch.clamp(target, min_value, max_value))
    return clip_loss + noclip_loss * eta


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
        return clamp_loss(input, target, self.module,
                          min_value=self.min_value, max_value=self.max_value, eta=self.eta)
