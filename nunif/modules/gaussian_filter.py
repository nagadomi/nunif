import torch
import torch.nn as nn
import torch.nn.functional as F
from .replication_pad2d import ReplicationPad2dNaive, ReplicationPad1dNaive
from .permute import kernel2d_to_conv2d_weight, kernel1d_to_conv1d_weight


def get_gaussian_kernel1d(kernel_size, dtype=None, device=None, sigma=None):
    if kernel_size == 1:
        return torch.ones((kernel_size,), dtype=dtype, device=device)

    # gaussin kernel formula is from torchvision
    if sigma is None:
        sigma = kernel_size * 0.15 + 0.35
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
    gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel


def get_gaussian_kernel2d(kernel_size, dtype=None, device=None, sigma=None):
    if isinstance(kernel_size, (int,)):
        kernel_size = [kernel_size, kernel_size]

    kernel1d_y = get_gaussian_kernel1d(kernel_size[0], sigma=sigma, dtype=dtype, device=device)
    kernel1d_x = get_gaussian_kernel1d(kernel_size[1], sigma=sigma, dtype=dtype, device=device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


class GaussianFilter2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None, sigma=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel2d(kernel_size, sigma=sigma)
            kernel = kernel2d_to_conv2d_weight(in_channels, kernel)
        self.register_buffer("kernel", kernel, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 4
            self.pad = ReplicationPad2dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        return F.conv2d(self.pad(x), weight=self.kernel, bias=None, groups=self.kernel.shape[0])


class SeparableGaussianFilter2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None, sigma=None):
        super().__init__()
        with torch.no_grad():
            kernel_h = get_gaussian_kernel2d([1, kernel_size], sigma=sigma)
            kernel_v = get_gaussian_kernel2d([kernel_size, 1], sigma=sigma)
            kernel_h = kernel2d_to_conv2d_weight(in_channels, kernel_h)
            kernel_v = kernel2d_to_conv2d_weight(in_channels, kernel_v)
        self.register_buffer("kernel_h", kernel_h, persistent=False)
        self.register_buffer("kernel_v", kernel_v, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 4
            self.pad = ReplicationPad2dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, weight=self.kernel_h, bias=None, groups=self.kernel_h.shape[0])
        x = F.conv2d(x, weight=self.kernel_v, bias=None, groups=self.kernel_v.shape[0])

        return x


class GaussianFilter1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel1d(kernel_size)
            kernel = kernel1d_to_conv1d_weight(in_channels, kernel)
        self.register_buffer("kernel", kernel, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 2
            self.pad = ReplicationPad1dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        return F.conv1d(self.pad(x), weight=self.kernel, bias=None, groups=self.kernel.shape[0])


def _test():
    print(get_gaussian_kernel1d(3))
    print(get_gaussian_kernel2d(3))

    blur = GaussianFilter2d(3, kernel_size=3, padding=1)
    x = torch.rand((4, 3, 8, 8))
    print(blur(x).shape)

    blur = GaussianFilter1d(3, kernel_size=3, padding=1)
    x = torch.rand((4, 3, 8))
    print(blur(x).shape)


def _test_sep():
    blur1 = GaussianFilter2d(3, kernel_size=7, padding=3)
    blur2 = SeparableGaussianFilter2d(3, kernel_size=7, padding=3)

    x = torch.rand((4, 3, 64, 64))

    z1 = blur1(x)
    z2 = blur2(x)

    print("diff GaussianFilter2d - SeparableGaussianFilter2d",
          (z1 - z2).abs().mean())


def _test_vis():
    import torchvision.io as io
    import torchvision.transforms.functional as TF

    blur = GaussianFilter2d(3, kernel_size=7, padding=3)
    x = (io.read_image("cc0/320/dog.png") / 256).unsqueeze(0)
    x = blur(x)
    TF.to_pil_image(x[0]).show()


def _bench():
    import time

    """
    ** kernel_size = 3
      GaussianFilter2d 2139.26 FPS
      SeparableGaussianFilter2d 1434.91 FPS
    ** kernel_size = 5
      GaussianFilter2d 1680.86 FPS
      SeparableGaussianFilter2d 1280.52 FPS
    ** kernel_size = 7
      GaussianFilter2d 1132.32 FPS
      SeparableGaussianFilter2d 1162.45 FPS
    ** kernel_size = 11
      GaussianFilter2d 665.44 FPS
      SeparableGaussianFilter2d 965.37 FPS
    ** kernel_size = 15
      GaussianFilter2d 414.36 FPS
      SeparableGaussianFilter2d 833.54 FPS
    ** kernel_size = 31
      GaussianFilter2d 114.03 FPS
      SeparableGaussianFilter2d 531.69 FPS
    """

    N = 100
    B = 4
    x = torch.rand((B, 3, 1080, 1920)).cuda()
    for kernel_size in (3, 5, 7, 11, 15, 31):
        blur1 = GaussianFilter2d(3, kernel_size=kernel_size, padding=kernel_size // 2).cuda()
        blur2 = SeparableGaussianFilter2d(3, kernel_size=kernel_size, padding=kernel_size // 2).cuda()

        print(f"** kernel_size = {kernel_size}")
        with torch.inference_mode():
            for _ in range(2):
                blur1(x)
                blur2(x)

            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(N):
                blur1(x)
            torch.cuda.synchronize()
            print("  GaussianFilter2d", round((N * B) / (time.perf_counter() - t), 2), "FPS")

            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(N):
                blur2(x)
            torch.cuda.synchronize()
            print("  SeparableGaussianFilter2d", round((N * B) / (time.perf_counter() - t), 2), "FPS")


if __name__ == "__main__":
    _test()
    _test_sep()
    _bench()
    _test_vis()
