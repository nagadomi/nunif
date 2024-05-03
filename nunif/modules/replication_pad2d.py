# MPS backend does not support nn.ReplicationPad2d
# use this instead
import torch
import torch.nn as nn


class ReplicationPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            # MPS or CPU on macOS
            self.pad = ReplicationPad2dNaive(padding)
        else:
            self.pad = nn.ReplicationPad2d(padding)

    def forward(self, x):
        return self.pad(x)


def replication_pad2d_naive(x, padding):
    assert x.ndim == 4 and len(padding) == 4
    left, right, top, bottom = padding
    if left > 0:
        x = torch.cat((*((x[:, :, :, :1].detach(),) * left), x), dim=3)
    if right > 0:
        x = torch.cat((x, *((x[:, :, :, -1:].detach(),) * right)), dim=3)
    if top > 0:
        x = torch.cat((*((x[:, :, :1, :].detach(),) * top), x), dim=2)
    if bottom > 0:
        x = torch.cat((x, *((x[:, :, -1:, :].detach(),) * bottom)), dim=2)
    return x


class ReplicationPad2dNaive(nn.Module):
    def __init__(self, padding):
        super().__init__()
        assert isinstance(padding, (list, tuple)) and len(padding) == 4
        self.padding = padding

    def forward(self, x):
        return replication_pad2d_naive(x, self.padding)


def _test():
    padding = (1, 2, 3, 4)
    my_pad = ReplicationPad2dNaive(padding)
    nn_pad = nn.ReplicationPad2d(padding)

    x = torch.rand((4, 3, 8, 8))
    y1 = my_pad(x)
    y2 = nn_pad(x)
    assert (y1 - y2).abs().sum() == 0

    for _ in range(100):
        padding = torch.randint(0, 10, (4,)).tolist()
        my_pad = ReplicationPad2dNaive(padding).cuda()
        nn_pad = nn.ReplicationPad2d(padding).cuda()
        x = torch.rand((4, 3, 8, 8)).cuda()
        y1 = my_pad(x)
        y2 = nn_pad(x)
        assert (y1 - y2).abs().sum() == 0


if __name__ == "__main__":
    _test()
