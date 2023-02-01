from torch import nn
import torch


def charbonnier_loss(input, target, reduction="mean", eps=1.0e-6):
    loss = torch.sqrt(((input - target) ** 2) + eps ** 2)
    if reduction is None or reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        return charbonnier_loss(input, target, eps=self.eps, reduction=self.reduction)
