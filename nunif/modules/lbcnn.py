import torch
import torch.nn as nn
import torch.nn.functional as F


""" ref: Local Binary Convolutional Neural Network
         https://arxiv.org/abs/1608.06049
"""


# training not tested 


def generate_lbcnn_filters(size, sparcity=0.9, seed=71):
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        filters = torch.bernoulli(torch.torch.full(size, 0.5)).mul_(2).add(-1)
        filters[torch.rand(filters.shape) > sparcity] = 0
    finally:
        torch.random.set_rng_state(rng_state)
    # print(filters)

    return filters


class RandomBinaryConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparcity=0.9, seed=None):
        super().__init__()
        self.stride = stride
        if padding == 0:
            self.pad = nn.Identity()
        else:
            self.pad = nn.ReplicationPad2d(padding)

        if seed is None:
            seed = torch.randint(0, 0x7fffffff, (1,)).item()
        self.register_buffer(
            "kernel",
            generate_lbcnn_filters((out_channels, in_channels, kernel_size, kernel_size),
                                   sparcity=sparcity, seed=seed))

    def forward(self, x):
        return F.conv2d(self.pad(x), weight=self.kernel, bias=None, stride=self.stride, padding=0)


def _test():
    x = torch.randn((32, 3, 128, 128))
    net = nn.Sequential(
        RandomBinaryConvolution(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        RandomBinaryConvolution(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        RandomBinaryConvolution(64, 64, kernel_size=3, padding=0),
        nn.BatchNorm2d(64))

    out = net(x)
    print(out.shape, out.min(), out.max(), out.mean())


if __name__ == "__main__":
    _test()
