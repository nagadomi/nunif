import torch
import torch.nn as nn
import torch.nn.functional as F
from .softpool import soft_pool2d


class MultiscaleLoss(nn.Module):
    """ Wrapper Module for `(loss(x, y) * w1 + loss(downscale(x), downscale(y)) * w2 ..`
    """
    def __init__(self, module, scale_factors=(1, 2), weights=(0.8, 0.2), mode="bilinear", antialias=None, align_corners=None):
        super().__init__()
        assert len(scale_factors) == len(weights)
        assert mode in {"bicubic", "bilinear", "avg", "softpool"}
        self.module = module
        self.scale_factors = scale_factors
        self.weights = weights
        self.mode = mode
        self.antialias = antialias
        self.align_corners = align_corners

    def forward(self, input, target):
        loss = 0
        for scale_factor, weight in zip(self.scale_factors, self.weights):
            if scale_factor == 1:
                x = input
                t = target
            else:
                if self.mode == "avg":
                    x = F.avg_pool2d(input, scale_factor)
                    t = F.avg_pool2d(target, scale_factor)
                elif self.mode == "softpool":
                    x = soft_pool2d(input, scale_factor)
                    t = soft_pool2d(target, scale_factor)
                else:
                    x = F.interpolate(input, scale_factor=1.0 / scale_factor,
                                      mode=self.mode, antialias=self.antialias, align_corners=self.align_corners)
                    t = F.interpolate(target, scale_factor=1.0 / scale_factor,
                                      mode=self.mode, antialias=self.antialias, align_corners=self.align_corners)
            loss = loss + self.module(x, t) * weight
        return loss


def _test():
    x = torch.randn((4, 3, 32, 32))
    t = torch.randn((4, 3, 32, 32))
    criterion = MultiscaleLoss(nn.L1Loss())
    print(criterion(x, t))

    criterion = MultiscaleLoss(nn.L1Loss(), mode="avg")
    print(criterion(x, t))
    criterion = MultiscaleLoss(nn.L1Loss(), mode="softpool")
    print(criterion(x, t))


if __name__ == "__main__":
    _test()
