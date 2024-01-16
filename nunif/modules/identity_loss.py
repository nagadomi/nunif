from torch import nn


class IdentityLoss(nn.Module):
    def forward(self, input, target):
        return input
