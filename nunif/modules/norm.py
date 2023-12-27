import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Normalize(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2., dim=self.dim, eps=self.eps)


class LayerNormNoBias(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight)


class LayerNormNoBias2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((dim,)))

    def forward(self, x):
        x = F.group_norm(x, num_groups=1, weight=self.weight, bias=None)
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


if __name__ == "__main__":
    # _test_frn()
    # _test_l2norm()
    _test_layer_norm()
