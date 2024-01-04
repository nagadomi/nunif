import torch
import torch.nn as nn
import torch.nn.functional as F
from .permute import bchw_to_bhwc, bhwc_to_bchw


class L2Normalize(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2., dim=self.dim, eps=self.eps)


def LayerNormNoBias(normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
    # bias=False, requires pytorch 2.1
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                        bias=False,
                        device=device, dtype=dtype)


class LayerNormNoBias2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormNoBias(dim)

    def forward(self, x):
        x = bhwc_to_bchw(self.norm(bchw_to_bhwc(x)))
        return x


class GroupNormNoBias(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        else:
            self.weight = self.register_parameter("weight", None)

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, weight=self.weight, bias=None, eps=self.eps)
        return x


def _test_l2norm():
    x = torch.randn((1, 4, 2, 2))
    model = L2Normalize()
    y = model(x)
    print("dim=1")
    print(x)
    print(y.shape, y)
    print(torch.sqrt(torch.sum(y ** 2, dim=1, keepdim=True)))

    x = torch.randn((1, 2, 2, 4))
    model = L2Normalize(dim=3)
    y = model(x)
    print("dim=3")
    print(x)
    print(y.shape, y)
    print(torch.sqrt(torch.sum(y ** 2, dim=3, keepdim=True)))


def _test_layer_norm():
    print(LayerNormNoBias(4)(torch.zeros((1, 2, 2, 4))).shape)
    print(GroupNormNoBias(1, 4)(torch.zeros((1, 4, 2, 2))).shape)
    print(GroupNormNoBias(1, 4, affine=False)(torch.zeros((1, 4, 2, 2))).shape)
    print(LayerNormNoBias2d(4)(torch.zeros((1, 4, 2, 2))).shape)


if __name__ == "__main__":
    # _test_l2norm()
    _test_layer_norm()
