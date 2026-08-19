import torch
import torch.nn as nn
import torch.nn.functional as F


def align_up(n, factor=1.5, mod=32):
    n = max(int(n * factor), mod)
    rem = n % mod
    if rem != 0:
        return n + (mod - rem)
    return n


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert x.shape[dim] % 2 == 0
    x1, x2 = x.chunk(2, dim=dim)
    # DINOv2 order; differs from F.glu
    return F.silu(x1) * x2


class SwiGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, dim=self.dim)


def _test():
    dim = 32
    dim2 = align_up(dim, factor=1.5)

    model = nn.Sequential(
        nn.Linear(dim, dim2 * 2),
        SwiGLU(),
        nn.Linear(dim2, dim),
    ).cuda()
    x = torch.zeros((4, 32)).cuda()
    model(x).sum().backward()

    for i in range(0, 64):
        n = align_up(i, factor=2, mod=32)
        assert n >= 32 and n % 32 == 0


if __name__ == "__main__":
    _test()
