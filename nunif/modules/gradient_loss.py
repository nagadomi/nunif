import torch
import torch.nn as nn
import torch.nn.functional as F
from . weighted_loss import WeightedLoss
from . clamp_loss import ClampLoss
from . color import RGBToYRGB


def gradient(x, diag=False):
    B, C, H, W = x.shape

    # [A B E]
    # [C D _]

    # D - B
    y_grad = x[:, :, 1:, 1:] - x[:, :, :-1, 1:].detach()
    # D - C
    x_grad = x[:, :, 1:, 1:] - x[:, :, 1:, :-1].detach()
    if not diag:
        return y_grad, x_grad
    else:
        # D - A
        diag1_grad = x[:, :, 1:, 1:] - x[:, :, :-1, :-1].detach()
        # D - E
        diag2_grad = x[:, :, 1:, 1:-1] - x[:, :, :-1, 2:].detach()
        return y_grad, x_grad, diag1_grad, diag2_grad


def gradient_loss(input, target, diag=False, loss_fn=F.l1_loss):
    input_grads = gradient(input, diag=diag)
    target_grads = gradient(target, diag=diag)
    return sum(loss_fn(ig, tg) for ig, tg in zip(input_grads, target_grads)) / len(input_grads)


class GradientLoss(nn.Module):
    def __init__(self, diag=False, loss_fn=F.l1_loss):
        super().__init__()
        self.loss_fn = loss_fn
        self.diag = diag

    def forward(self, input, target):
        return gradient_loss(input, target, self.diag, self.loss_fn)


def L1GradientLoss(weight=1.0, diag=False):
    return WeightedLoss((nn.L1Loss(), GradientLoss(diag=diag)), (1.0, weight))


def YRGBL1GradientLoss(weight=1.0, diag=False):
    return WeightedLoss((ClampLoss(nn.L1Loss()), ClampLoss(GradientLoss(diag=diag))),
                        weights=(1.0, weight), preprocess=RGBToYRGB())


if __name__ == "__main__":
    input = torch.randn((4, 32, 4, 4))
    target = torch.randn((4, 32, 4, 4))
    loss1 = GradientLoss()
    loss2 = L1GradientLoss()
    loss3 = YRGBL1GradientLoss()
    print(loss1(input, target))
    print(loss2(input, target))
    print(loss3(input, target))
