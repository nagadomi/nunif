import torch


def inplace_clip(x, min_value, max_value):
    return torch.clamp_(x, min_value, max_value)
