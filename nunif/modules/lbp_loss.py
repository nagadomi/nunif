import torch
from torch import nn
from .charbonnier_loss import CharbonnierLoss
from .clamp_loss import ClampLoss
from .channel_weighted_loss import LuminanceWeightedLoss, AverageWeightedLoss


def generate_lbcnn_filters(size, sparcity=0.9, seed=71):
    """ from Local Binary Convolutional Neural Network
    """
    out_channels, in_channels, kernel_size, _ = size
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        filters = torch.bernoulli(torch.torch.full(size, 0.5)).mul_(2).add(-1)
        filters[torch.rand(filters.shape) > sparcity] = 0
    finally:
        torch.random.set_rng_state(rng_state)
    # print(filters)

    return filters


def generate_random_filters(size, sparcity=0.5):
    """ normalized random filter
    output = [-0.5, 0.5], abs(diff) = [0, 1]
    """
    out_channels, in_channels, kernel_size, _ = size
    filters = torch.bernoulli(torch.torch.full(size, sparcity))
    filters[:, :, kernel_size // 2, kernel_size // 2] = 0
    filter_sum = filters.view(out_channels, in_channels, -1).sum(dim=2).add_(1e-6)
    filter_sum = filter_sum.view(out_channels, in_channels, 1, 1).expand(size=filters.shape)
    filters.div_(filter_sum)
    filters[:, :, kernel_size // 2, kernel_size // 2] = -1
    filters.mul_(0.5)
    #  print(filters)

    return filters


"""
Note: Be careful not to initialize by `if isinstance(module, nn.Conv2d):` condition
"""


class RandomBinaryConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparcity=0.9, seed=71):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False)
        self.conv.weight.data.copy_(generate_lbcnn_filters(self.conv.weight.data.shape, sparcity, seed=seed))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class RandomFilterConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparcity=0.5):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)
        self.conv.weight.data.copy_(generate_random_filters(self.conv.weight.data.shape, sparcity))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class LBPLoss(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, sparcity=0.9, loss=None, seed=71):
        super().__init__()
        self.conv = RandomBinaryConvolution(in_channels, out_channels - out_channels % in_channels,
                                            kernel_size=kernel_size, padding=0,
                                            sparcity=sparcity, seed=seed)
        if loss is None:
            self.loss = CharbonnierLoss()
        else:
            self.loss = loss

        # [0] = identity filter
        self.conv.conv.weight.data[0] = 0
        self.conv.conv.weight.data[0, :, kernel_size // 2, kernel_size // 2] = 0.5 * kernel_size ** 2

    def forward(self, input, target):
        b, ch, *_ = input.shape
        return self.loss(self.conv(input), self.conv(target))


def YLBP(kernel_size=3):
    return ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size)))


def RGBLBP(kernel_size=3):
    return ClampLoss(AverageWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size),
                                         in_channels=3))


class YL1LBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = YLBP(kernel_size=kernel_size)
        self.l1 = ClampLoss(LuminanceWeightedLoss(torch.nn.L1Loss()))
        self.weight = weight

    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        l1_loss = self.l1(input, target)
        return l1_loss + lbp_loss * self.weight


class L1LBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = RGBLBP(kernel_size=kernel_size)
        self.l1 = ClampLoss(AverageWeightedLoss(torch.nn.L1Loss(), in_channels=3))
        self.weight = weight

    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        l1_loss = self.l1(input, target)
        return l1_loss + lbp_loss * self.weight


def _check_gradient_norm():
    l1_loss = nn.L1Loss()
    lbp_loss = RGBLBP()

    x1 = torch.ones((1, 3, 32, 32), requires_grad=True) / 2.
    x2 = torch.ones((1, 3, 32, 32), requires_grad=True) / 2.
    x2 = x2 + torch.randn(x2.shape) * 0.01

    loss1 = l1_loss(x1, x2)
    loss2 = lbp_loss(x1, x2)

    grad1 = torch.autograd.grad(loss1, x2, retain_graph=True)[0]
    grad2 = torch.autograd.grad(loss2, x2, retain_graph=True)[0]
    norm1 = torch.norm(grad1, p=2)
    norm2 = torch.norm(grad2, p=2)

    # norm1 / norm2 = around 0.41
    print(norm1, norm2, norm1 / norm2)


def _test():
    conv = RandomFilterConvolution(1, 8, 3, 1)
    data = torch.rand((4, 1, 16, 16))
    print(conv(data).shape)

    conv = RandomBinaryConvolution(1, 8, 3, 1)
    data = torch.rand((4, 1, 16, 16))
    print(conv(data).shape)


if __name__ == "__main__":
    _check_gradient_norm()
