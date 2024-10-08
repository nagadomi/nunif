import torch
import torch.nn as nn
import torch.nn.functional as F
from .lbcnn import generate_lbcnn_filters
from .charbonnier_loss import CharbonnierLoss
from .clamp_loss import ClampLoss
from .channel_weighted_loss import LuminanceWeightedLoss, AverageWeightedLoss
from .compile_wrapper import conditional_compile
from .color import rgb_to_yrgb


def generate_lbp_kernel(in_channels, out_channels, kernel_size=3, seed=71):
    kernel = generate_lbcnn_filters((out_channels, in_channels, kernel_size, kernel_size), seed=seed)
    # [0] = identity filter
    kernel[0] = 0
    kernel[0, :, kernel_size // 2, kernel_size // 2] = 0.5 * kernel_size ** 2
    return kernel


class LBPLoss(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, loss=None, seed=71):
        super().__init__()
        self.groups = in_channels
        self.register_buffer(
            "kernel",
            generate_lbp_kernel(in_channels, out_channels - out_channels % in_channels,
                                kernel_size, seed=seed))
        if loss is None:
            self.loss = CharbonnierLoss()
        else:
            self.loss = loss

    def conv(self, x):
        return F.conv2d(x, weight=self.kernel, bias=None, stride=1, padding=0, groups=self.groups)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        b, ch, *_ = input.shape
        return self.loss(self.conv(input), self.conv(target))


def YLBP(kernel_size=3, out_channels=64):
    return ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size, out_channels=out_channels)),
                     clamp_l1=True)


def RGBLBP(kernel_size=3):
    return ClampLoss(AverageWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size),
                                         in_channels=3), clamp_l1=True)


class YRGBLBP(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.loss = ClampLoss(AverageWeightedLoss(LBPLoss(in_channels=1, kernel_size=kernel_size), in_channels=4),
                              clamp_l1=True)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        input = rgb_to_yrgb(input, y_clamp=True)
        target = rgb_to_yrgb(target, y_clamp=True)
        return self.loss(input, target)


class YRGBL1LBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = YRGBLBP(kernel_size=kernel_size)
        self.l1 = ClampLoss(LuminanceWeightedLoss(torch.nn.L1Loss()))
        self.weight = weight

    @conditional_compile("NUNIF_TRAIN")
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


def _test_clamp_input_only():
    import time

    lbp_old = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=3))).cuda()
    lbp_new = YLBP().cuda()

    torch.manual_seed(71)
    x = torch.randn((4, 3, 256, 256)) / 2 + 0.5
    y = torch.clamp(x + (torch.randn((4, 3, 256, 256)) / 10), 0, 1)
    x = x.cuda()
    y = y.cuda()
    print("diff", (lbp_old(x, y) - lbp_new(x, y)).abs())

    N = 100
    t = time.perf_counter()
    for _ in range(N):
        lbp_old(x, y)
    torch.cuda.synchronize()
    print("ylbp_old", time.perf_counter() - t)
    t = time.perf_counter()
    for _ in range(N):
        lbp_new(x, y)
    torch.cuda.synchronize()
    print("ylbp_new", time.perf_counter() - t)


if __name__ == "__main__":
    # _check_gradient_norm()
    _test_clamp_input_only()
