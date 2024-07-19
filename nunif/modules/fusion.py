import torch
import torch.nn as nn
import torch.nn.functional as F


class Lerp(nn.Module):
    def __init__(self, weight_shape=None):
        super().__init__()
        if weight_shape is None:
            weight_shape = 1
        if torch.is_tensor(weight_shape):
            self.weight = nn.Parameter(weight_shape.detach().clone())
        else:
            self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=torch.float32))

    def forward(self, input, end):
        # out = input + (0. 5 + self.weight) * (end - start)
        return torch.lerp(input, end, (0.5 + self.weight).to(input.dtype))


class AdaptiveWeight(nn.Module):
    def __init__(self, n=2):
        assert n >= 2
        super().__init__()
        self.weight = nn.Parameter(torch.ones((n,), dtype=torch.float32))

    def forward(self):
        # exp(x_i) / sum([exp(x_j) for x_j in x])
        return F.softmax(self.weight, dim=0)


class AdaptiveWeightedAdd(nn.Module):
    def __init__(self, in_channels, n=2):
        super().__init__()
        self.adaptive_weight = AdaptiveWeight(n=n)

    def forward(self, *inputs):
        weight = self.adaptive_weight()
        # print(inputs[0].shape[1], weight)

        assert weight.numel() == len(inputs)
        return sum([x * weight[i] for i, x in enumerate(inputs)])


if __name__ == "__main__":
    avg = AdaptiveWeightedAdd(32, 4)
    x = torch.ones((4, 32, 4, 4))
    x = avg(x, x, x, x)
    assert x.sum() == x.numel()
