from torch import nn


class ChannelWeightedLoss(nn.Module):
    """ Wrapper Module for channel weight
    """
    def __init__(self, module, weight):
        super().__init__()
        self.module = module
        self.weight = weight

    def forward(self, input, target):
        b, ch, *_ = input.shape
        assert (ch == len(self.weight))
        return sum([self.module(input[:, i:i + 1, :, :], target[:, i:i + 1, :, :]) * self.weight[i]
                    for i in range(ch)])


LUMINANCE_WEIGHT = [0.29891, 0.58661, 0.11448]


class LuminanceWeightedLoss(ChannelWeightedLoss):
    def __init__(self, module):
        super().__init__(module, weight=LUMINANCE_WEIGHT)


class AverageWeightedLoss(ChannelWeightedLoss):
    def __init__(self, module, in_channels=3):
        weight = [1.0 / in_channels] * in_channels
        super().__init__(module, weight=weight)
