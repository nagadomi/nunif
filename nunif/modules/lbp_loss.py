import torch
from torch import nn


def generate_lbcnn_filters(size, sparcity=0.9):
    """ from Local Binary Convolutional Neural Network
    """
    out_channels, in_channels, kernel_size, _ = size
    filters = torch.bernoulli(torch.torch.full(size, 0.5)).mul_(2).add(-1)
    filters[torch.rand(filters.shape) > sparcity] = 0
    # print(filters)

    return filters


def generate_random_filters(size, sparcity=0.5):
    """ normalized random filter
    output = [-0.5, 0.5], abs(diff) = [0, 1]
    """
    out_channels, in_channels, kernel_size, _ = size
    filters = torch.bernoulli(torch.torch.full(size, sparcity))
    filters[:, :, kernel_size//2, kernel_size//2] = 0
    filter_sum = filters.view(out_channels, in_channels, -1).sum(dim=2).add_(1e-6)
    filter_sum = filter_sum.view(out_channels, in_channels, 1, 1).expand(size=filters.shape)
    filters.div_(filter_sum)
    filters[:, :, kernel_size//2, kernel_size//2] = -1
    filters.mul_(0.5)
    #  print(filters)

    return filters


"""
Note: Be careful not to initialize by `if isinstance(module, nn.Conv2d):` condition
"""

class RandomBinaryConvolution(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparcity=0.9):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False)
        self.weight.data.copy_(generate_lbcnn_filters(self.weight.data.shape, sparcity))
        self.weight.requires_grad_(False)


class RandomFilterConvolution(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparcity=0.5):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)
        self.weight.data.copy_(generate_random_filters(self.weight.data.shape, sparcity))
        self.weight.requires_grad_(False)


class LBPLoss(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, delta=1.0, sparcity=0.9):
        super().__init__()
        self.conv = RandomBinaryConvolution(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=0,
                                            sparcity=sparcity)
        self.loss = nn.HuberLoss(delta=delta)

        # [0] = identity filter
        self.conv.weight.data[0] = 0
        self.conv.weight.data[0, :, kernel_size//2, kernel_size//2] = 0.5 * kernel_size **2

    def forward(self, input, target):
        b, ch, *_ = input.shape
        return self.loss(self.conv(input), self.conv(target))


if __name__ == "__main__":
    conv = RandomFilterConvolution(1, 8, 3, 1)
    data = torch.rand((4, 1, 16, 16))
    print(conv(data).shape)

    conv = RandomBinaryConvolution(1, 8, 3, 1)
    data = torch.rand((4, 1, 16, 16))
    print(conv(data).shape)
