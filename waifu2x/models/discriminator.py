import torch
from .u3c import U3ConditionalDiscriminator
from .l3v1c import L3V1ConditionalDiscriminator


def _test():
    l3v1c = L3V1ConditionalDiscriminator()
    u3c = U3ConditionalDiscriminator()

    S = 64 * 4 - 38 * 2
    x = torch.zeros((1, 3, S, S))
    c = torch.zeros((1, 3, S, S))
    print(u3c(x, c, 4).shape)
    print([z.shape for z in l3v1c(x, c, 4)])


if __name__ == "__main__":
    _test()
