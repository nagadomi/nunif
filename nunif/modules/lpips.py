import torch
from torch import nn
import lpips


class LPIPSWith(nn.Module):
    def __init__(self, base_loss, base_loss_weight=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.base_loss_weight = base_loss_weight
        self.lpips = lpips.LPIPS(net='vgg')
        self.lpips.requires_grad_(False)

    def forward(self, input, target):
        base_loss = self.base_loss(input, target)
        lpips_loss = self.lpips(input, target, normalize=True).mean()
        return base_loss * self.base_loss_weight + lpips_loss


def _test():
    loss = LPIPSWith(nn.L1Loss())
    print(loss.lpips)
    print([p.requires_grad for p in loss.lpips.parameters()])

    x = torch.randn((4, 3, 64, 64))
    t = torch.randn((4, 3, 64, 64))
    print(loss(x, t))


if __name__ == "__main__":
    _test()
