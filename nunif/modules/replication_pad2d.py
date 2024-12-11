# 1. MPS backend does not support nn.ReplicationPad2d. Use this instead.
# 2. ReplicationPad2 calculates the gradient of the padding values multiply.
#    This implementation can use `detach=True` option.

import torch
import torch.nn as nn
import sys


class ReplicationPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if sys.platform == "darwin":
            # macOS, detach=False for compatibility
            self.pad = ReplicationPad2dNaive(padding, detach=False)
        else:
            self.pad = nn.ReplicationPad2d(padding)

    def forward(self, x):
        return self.pad(x)


def replication_pad2d_naive(x, padding, detach=False):
    assert x.ndim == 4 and len(padding) == 4
    left, right, top, bottom = padding

    detach_fn = lambda t: t.detach() if detach else t
    if left > 0:
        x = torch.cat((*((detach_fn(x[:, :, :, :1]),) * left), x), dim=3)
    elif left < 0:
        x = x[:, :, :, -left:]
    if right > 0:
        x = torch.cat((x, *((detach_fn(x[:, :, :, -1:]),) * right)), dim=3)
    elif right < 0:
        x = x[:, :, :, :right]
    if top > 0:
        x = torch.cat((*((detach_fn(x[:, :, :1, :]),) * top), x), dim=2)
    elif top < 0:
        x = x[:, :, -top:, :]
    if bottom > 0:
        x = torch.cat((x, *((detach_fn(x[:, :, -1:, :]),) * bottom)), dim=2)
    elif bottom < 0:
        x = x[:, :, :bottom, :]

    return x.contiguous()


def replication_pad1d_naive(x, padding, detach=False):
    assert x.ndim == 3 and len(padding) == 2
    left, right = padding

    detach_fn = lambda t: t.detach() if detach else t
    if left > 0:
        x = torch.cat((*((detach_fn(x[:, :, :1]),) * left), x), dim=2)
    elif left < 0:
        x = x[:, :, -left:]
    if right > 0:
        x = torch.cat((x, *((detach_fn(x[:, :, -1:]),) * right)), dim=2)
    elif right < 0:
        x = x[:, :, :right]

    return x.contiguous()


class ReplicationPad2dNaive(nn.Module):
    def __init__(self, padding, detach=False):
        super().__init__()
        assert isinstance(padding, (list, tuple)) and len(padding) == 4
        self.padding = padding
        self.detach = detach

    def forward(self, x):
        return replication_pad2d_naive(x, self.padding, detach=self.detach)


class ReplicationPad1dNaive(nn.Module):
    def __init__(self, padding, detach=False):
        super().__init__()
        assert isinstance(padding, (list, tuple)) and len(padding) == 2
        self.padding = padding
        self.detach = detach

    def forward(self, x):
        return replication_pad1d_naive(x, self.padding, detach=self.detach)


def _test():
    padding = (1, 2, 3, 4)
    my_pad = ReplicationPad2dNaive(padding)
    nn_pad = nn.ReplicationPad2d(padding)

    x = torch.rand((4, 3, 8, 8))
    y1 = my_pad(x)
    y2 = nn_pad(x)
    assert (y1 - y2).abs().sum() == 0

    for _ in range(10):
        padding = torch.randint(-3, 3, (4,)).tolist()
        my_pad = ReplicationPad2dNaive(padding).cuda()
        nn_pad = nn.ReplicationPad2d(padding).cuda()
        x = torch.rand((4, 3, 8, 8)).cuda()
        y1 = my_pad(x)
        y2 = nn_pad(x)
        assert (y1 - y2).abs().sum() == 0


def _test_grad():
    # https://github.com/pytorch/pytorch/issues/68879
    conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
    x = torch.ones((1, 1, 6, 6), requires_grad=True)

    nn.init.constant_(conv.weight, 1)
    nn.init.constant_(conv.bias, 0)

    # all 1
    y = conv(replication_pad2d_naive(x, (1,) * 4, detach=True)).sum()
    y.backward()
    print(x.grad)
    x.grad.zero_()

    # grad of the padded values is multiply computed
    y = conv(replication_pad2d_naive(x, (1,) * 4, detach=False)).sum()
    y.backward()
    print(x.grad)


if __name__ == "__main__":
    _test()
    _test_grad()
