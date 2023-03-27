import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleLoss(nn.Module):
    """ Wrapper Module for `(loss(x, y) * w1 + loss(downscale(x), downscale(y)) * w2 ..`
    """
    def __init__(self, module, scale_factors=(1, 2), weights=(0.8, 0.2), mode="bicubic"):
        super().__init__()
        assert len(scale_factors) == len(weights)
        self.module = module
        self.scale_factors = scale_factors
        self.weights = weights
        self.mode = mode

    def forward(self, input, target):
        loss = 0
        for scale_factor, weight in zip(self.scale_factors, self.weights):
            if scale_factor == 1:
                x = input
                t = target
            else:
                x = F.interpolate(input, scale_factor=1.0 / scale_factor, mode=self.mode)
                t = F.interpolate(target, scale_factor=1.0 / scale_factor, mode=self.mode)
            loss = loss + self.module(x, t) * weight
        return loss


def _test():
    x = torch.randn((4, 3, 32, 32))
    t = torch.randn((4, 3, 32, 32))
    criterion = MultiscaleLoss(nn.L1Loss())
    print(criterion(x, t))


if __name__ == "__main__":
    _test()
