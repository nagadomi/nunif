import torch
import torch.nn as nn
import torch.nn.functional as F
from .lbcnn import generate_lbcnn_filters
from .charbonnier_loss import CharbonnierLoss
from .clamp_loss import ClampLoss
from .channel_weighted_loss import LuminanceWeightedLoss, AverageWeightedLoss
from .compile_wrapper import conditional_compile
from .color import rgb_to_yrgb
from .flat_color_loss import FlatColorWeightedLoss


def generate_lbp_kernel(in_channels, out_channels, kernel_size=3, seed=71):
    with torch.no_grad():
        kernel = generate_lbcnn_filters((out_channels, in_channels, kernel_size, kernel_size), seed=seed)
        # [0] = identity filter
        kernel[0] = 0
        kernel[0, :, kernel_size // 2, kernel_size // 2] = 0.5 * kernel_size ** 2
        kernel = kernel / kernel_size
        return kernel


class LBPLoss(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, loss=None, seed=71, num_kernels=32):
        super().__init__()
        self.groups = in_channels
        self.num_kernels = num_kernels
        kernels = torch.stack([
            generate_lbp_kernel(in_channels, out_channels - out_channels % in_channels,
                                kernel_size, seed=seed + i)
            for i in range(num_kernels)])
        self.register_buffer("kernels", kernels)
        if loss is None:
            self.loss = CharbonnierLoss()
        else:
            self.loss = loss

    @conditional_compile("NUNIF_TRAIN")
    def conv(self, x, i):
        return F.conv2d(x, weight=self.kernels[i], bias=None, stride=1, padding=0, groups=self.groups)

    def forward(self, input, target):
        b, ch, *_ = input.shape

        if self.training:
            i = torch.randint(low=0, high=self.num_kernels, size=(1,)).item()
        else:
            i = 0

        return self.loss(self.conv(input, i), self.conv(target, i))


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
        self.l1 = ClampLoss(torch.nn.L1Loss())
        self.weight = weight

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        l1_loss = self.l1(input, target)
        return l1_loss + lbp_loss * self.weight


class YRGBFlatLBP(nn.Module):
    def __init__(self, kernel_size=5, weight=0.4):
        super().__init__()
        self.lbp = YRGBLBP(kernel_size=kernel_size)
        self.flat_l1l2 = ClampLoss(FlatColorWeightedLoss())
        self.weight = weight

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, input, target):
        lbp_loss = self.lbp(input, target)
        flat_loss = self.flat_l1l2(input, target)
        return flat_loss + lbp_loss * self.weight


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
