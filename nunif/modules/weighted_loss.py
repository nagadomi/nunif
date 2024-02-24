import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """ Wrapper Module for `loss(x, y) * w1 + loss2(x, y) * w2, ...`
    """
    def __init__(self, modules, weights, preprocess=None):
        super().__init__()
        assert len(modules) == len(weights)
        self.losses = nn.ModuleList(modules)
        self.weights = weights
        self.preprocess = nn.Identity() if preprocess is None else preprocess

    def forward(self, input, target):
        loss = 0.0
        input = self.preprocess(input)
        target = self.preprocess(target)
        for module, weight in zip(self.losses, self.weights):
            if weight == 1.0:
                loss = loss + module(input, target)
            else:
                loss = loss + module(input, target) * weight
        return loss


def _test():
    x = torch.randn((4, 3, 32, 32))
    t = torch.randn((4, 3, 32, 32))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    multi = WeightedLoss((nn.L1Loss(), nn.MSELoss()), (1.0, 0.1))
    assert torch.isclose(l1(x, t) + mse(x, t) * 0.1, multi(x, t))


if __name__ == "__main__":
    _test()
