import torch
from torch import nn
import torch.nn.functional as F


def _basic_module_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    else:
        pass


def basic_module_init(model):
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
        _basic_module_init(model)
    else:
        for m in model.modules():
            _basic_module_init(m)


def icnr_init(m, scale_factor):
    with torch.no_grad():
        assert isinstance(m, nn.Conv2d)
        OUT, IN, H, W = m.weight.data.shape
        assert OUT % (scale_factor ** 2) == 0
        weight = torch.zeros((OUT // (scale_factor ** 2), IN, H, W))
        nn.init.kaiming_normal_(weight)
        if scale_factor > 1:
            weight = weight.permute(1, 0, 2, 3)
            weight = F.interpolate(weight, scale_factor=scale_factor, mode="nearest")
            weight = F.pixel_unshuffle(weight, scale_factor)
            weight = weight.permute(1, 0, 2, 3)
        m.weight.data.copy_(weight)
        nn.init.constant_(m.bias, 0)
