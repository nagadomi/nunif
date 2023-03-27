import torch
from torch import nn
import lpips
from os import path


# MODEL_PATH = None
# MODEL_PATH = path.join(path.dirname(__file__), "_lpips_1.pth")
MODEL_PATH = path.join(path.dirname(__file__), "_lpips_2.pth")


class LPIPSWith(nn.Module):
    def __init__(self, base_loss, weight=1.):
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.lpips = lpips.LPIPS(net='vgg', model_path=MODEL_PATH).eval()
        self.lpips.requires_grad_(False)

    def forward(self, input, target):
        base_loss = self.base_loss(input, target)
        lpips_loss = self.lpips(input, target, normalize=True).mean()
        return base_loss + lpips_loss * self.weight


def _test():
    loss = LPIPSWith(nn.L1Loss())
    print(loss.lpips)
    print(loss.lpips.training)
    print([p.requires_grad for p in loss.lpips.parameters()])

    x = torch.randn((4, 3, 64, 64))
    t = torch.randn((4, 3, 64, 64))
    print(loss(x, t))


if __name__ == "__main__":
    _test()
