from torch import nn
import torch


def charbonnier_loss(input, target, reduction="mean"):
    loss = torch.sqrt(((input - target) ** 2))
    if reduction is None or reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()


class CharbonnierLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return charbonnier_loss(input, target, reduction=self.reduction)
