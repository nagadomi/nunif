import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class EmptySEBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x


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


class SelfAttention2d(nn.Module):
    """ from Latent Diffusion
    """
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        z = self.norm(x)
        q, k, v = self.q(z), self.k(z), self.v(z)
        z = torch.bmm(q.view(b, c, h * w).permute(0, 2, 1),
                      k.view(b, c, h * w)) * math.sqrt(1 / c)
        z = F.softmax(z, dim=2)
        z = torch.bmm(v.view(b, c, h * w), z.permute(0, 2, 1)).view(b, c, h, w)
        z = self.proj(z)

        return x + z


def _spec():
    device = "cuda:0"
    x = torch.rand((4, 128, 32, 32)).to(device)
    b, c, h, w = x.shape

    sse = LocalSEBlock(c).to(device)
    print(sse(x).shape)

    awvavg = SelfWeightedAvgPool2d(c, 2).to(device)
    print(awvavg(x).shape)

    sa2 = SelfAttention2d(c).to(device)
    print(sa2(x).shape)


if __name__ == "__main__":
    _spec()
