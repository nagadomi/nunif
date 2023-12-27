import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SEBlock(nn.Module):
    """ from Squeeze-and-Excitation Networks
    """
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        b, c, _, _ = x.size()
        z = F.adaptive_avg_pool2d(x, 1)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)
        return x * z.expand(x.shape)


class SEBlockNHWC(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, in_channels // reduction, bias=bias)
        self.lin2 = nn.Linear(in_channels // reduction, in_channels, bias=bias)

    def forward(self, x):
        B, H, W, C = x.size()
        z = x.mean(dim=[1, 2], keepdim=True)
        z = F.relu(self.lin1(z), inplace=True)
        z = torch.sigmoid(self.lin2(z))
        return x * z


class SNSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias))

    def forward(self, x):
        b, c, _, _ = x.size()
        z = F.adaptive_avg_pool2d(x, 1)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)
        return x * z.expand(x.shape)


class LocalSEBlock(nn.Module):
    """
    SEBlock with fixed size local average pooling
    instead of variable size global average pooling.
    (in tiled rendering these are different)
    """
    def __init__(self, in_channels, kernel_size=16, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, h, w = x.size()
        assert (c == self.conv1.in_channels)
        assert (h >= self.kernel_size and h % self.kernel_size == 0)
        assert (w >= self.kernel_size and w % self.kernel_size == 0)

        z = F.avg_pool2d(x, kernel_size=(self.kernel_size, self.kernel_size))
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)

        return x * F.interpolate(z, size=(h, w), mode='nearest')


class AdaptiveSEBlock(nn.Module):
    """
    NOTE: if the input size is not a multiple of the output size,
    miss alignment may occur due to nearest upsampler.
    """
    def __init__(self, in_channels, output_size=(2, 2), reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)
        self.output_size = output_size

    def forward(self, x):
        b, c, h, w = x.size()
        assert (c == self.conv1.in_channels)

        z = F.adaptive_avg_pool2d(x, output_size=self.output_size)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)

        return x * F.interpolate(z, size=(h, w), mode='nearest')


class SelfWeightedAvgPool2d(nn.Module):
    """
    self weighted average pooling
    """
    def __init__(self, in_channels, kernel_size=2, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, stride=kernel_size,
                               padding=0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, kernel_size * kernel_size,
                               kernel_size=1, stride=1, padding=0, bias=bias)
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, h, w = x.size()
        assert (c == self.conv1.in_channels)
        assert (h >= self.kernel_size and h % self.kernel_size == 0)
        assert (w >= self.kernel_size and w % self.kernel_size == 0)

        # KxK spatial weight(sum=1) for KxKxC
        z = self.conv1(x)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = z.permute(0, 2, 3, 1)  # BCHW -> BHWC
        w = F.softmax(z, dim=3)
        w = w.permute(0, 3, 1, 2)  # BCHW
        w = F.pixel_shuffle(w, self.kernel_size)
        # weighted average pooling
        z = F.avg_pool2d(
            x * w.expand(x.shape),
            kernel_size=(self.kernel_size, self.kernel_size),
            divisor_override=1)
        return z


def _spec():
    device = "cuda:0"
    x = torch.rand((4, 128, 32, 32)).to(device)
    b, c, h, w = x.shape

    sse = LocalSEBlock(c).to(device)
    print(sse(x).shape)

    awvavg = SelfWeightedAvgPool2d(c, 2).to(device)
    print(awvavg(x).shape)


if __name__ == "__main__":
    _spec()
