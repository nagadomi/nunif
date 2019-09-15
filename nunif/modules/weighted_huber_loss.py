import torch
import torch.nn as nn
import torch.nn.functional as F
from . functional import weighted_huber_loss


class WeightedHuberLoss(nn.Module):
    def __init__(self, channel_weight, gamma=1, reduction='mean'):
        super(WeightedHuberLoss, self).__init__()
        self.weight = None
        self.channel_weight = channel_weight
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input, target):
        ch = input.shape[1]
        if self.weight is None:
            assert(self.channel_weight.shape[0] == ch)
            self.weight = channel_weight.view(ch, 1, 1).expand_as(input)

        return weighted_huber_loss(input, target, self.weight, gamma=self.gamma, reduction=self.reduction)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.nn.functional import smooth_l1_loss
    GAMMA = 0.5
    WEIGHT = 0.5
    t = torch.zeros((1, 1, 100)).float()
    y = torch.linspace(-2, 2, steps=100).view(1, 1, 100).float()
    channel_weight = torch.FloatTensor([WEIGHT])
    criterion = WeightedHuberLoss(channel_weight, gamma=GAMMA, reduction='none')
    loss = criterion.forward(y, t)
    loss_t = smooth_l1_loss(y, t, reduction='none')
    
    plt.plot(loss[0][0].numpy(), label="my huber loss")
    plt.plot(loss_t[0][0].numpy(), label="torch huber loss")
    plt.plot(((t - y) ** 2).view(100).numpy(), label="square loss")
    plt.plot(torch.abs(t - y).view(100).numpy(), label="abs loss")
    plt.legend()
    plt.show()
