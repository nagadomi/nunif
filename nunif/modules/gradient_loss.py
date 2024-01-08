import torch
import torch.nn as nn
import torch.nn.functional as F
from . weighted_loss import WeightedLoss


def gradient(x):
    B, C, H, W = x.shape
    return x[:, :, 1:, :] - x[:, :, :-1, :].detach(), x[:, :, :, 1:] - x[:, :, :, :-1].detach()


def gradient_loss(input, target, loss_fn=F.l1_loss):
    input_v, input_h = gradient(input)
    target_v, target_h = gradient(target)
    return (loss_fn(input_v, target_v) + loss_fn(input_h, target_h)) * 0.5


class GradientLoss(nn.Module):
    def __init__(self, loss_fn=F.l1_loss):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target):
        return gradient_loss(input, target, self.loss_fn)


def L1GradientLoss(weight=1.0):
    return WeightedLoss((nn.L1Loss(), GradientLoss()), (1.0, weight))


if __name__ == "__main__":
    input = torch.randn((4, 32, 4, 4))
    target = torch.randn((4, 32, 4, 4))
    loss1 = GradientLoss()
    loss2 = L1GradientLoss()
    print(loss1(input, target))
    print(loss2(input, target))
