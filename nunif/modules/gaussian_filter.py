import torch
import torch.nn as nn
import torch.nn.functional as F
from . replication_pad2d import ReplicationPad2dNaive, ReplicationPad1dNaive


def get_gaussian_kernel1d(kernel_size, dtype=None, device=None):
    # gaussin kernel formula is from torchvision
    sigma = kernel_size * 0.15 + 0.35
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
    gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel


def get_gaussian_kernel2d(kernel_size, dtype=None, device=None):
    if isinstance(kernel_size, (int,)):
        kernel_size = [kernel_size, kernel_size]

    kernel1d_y = get_gaussian_kernel1d(kernel_size[0], dtype=dtype, device=device)
    kernel1d_x = get_gaussian_kernel1d(kernel_size[1], dtype=dtype, device=device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


class GaussianFilter2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel2d(kernel_size)
            kernel = kernel.reshape(1, 1, *kernel.shape)
            kernel = kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()
        self.register_buffer("kernel", kernel)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 4
            self.pad = ReplicationPad2dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        return F.conv2d(self.pad(x), weight=self.kernel, bias=None, groups=self.kernel.shape[0])


class GaussianFilter1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel1d(kernel_size)
            kernel = kernel.reshape(1, 1, *kernel.shape)
            kernel = kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()
        self.register_buffer("kernel", kernel)

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


def _test_vis():
    import torchvision.io as io
    import torchvision.transforms.functional as TF

    blur = GaussianFilter2d(3, kernel_size=7, padding=3)
    x = (io.read_image("cc0/320/dog.png") / 256).unsqueeze(0)
    x = blur(x)
    TF.to_pil_image(x[0]).show()


if __name__ == "__main__":
    _test()
    _test_vis()
