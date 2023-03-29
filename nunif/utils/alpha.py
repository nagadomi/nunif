import torch
import torch.nn as nn


class ChannelWiseSum(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=1, padding=1, padding_mode="zeros", groups=in_channels,
                              bias=False)
        self.conv.weight.data.fill_(1.)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        if x.ndim == 3:
            # CHW
            x = x.unsqueeze(0)
            x = self.conv(x)
            x = x.squeeze(0)
        elif x.ndim == 2:
            # HW
            x = x.unsqueeze(0).unsqueeze(0)
            x = self.conv(x)
            x = x.squeeze(0).squeeze(0)
        else:
            # BCHW
            x = self.conv(x)

        return x


class AlphaBorderPadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum_alpha = ChannelWiseSum(1, 3)
        self.sum_rgb = ChannelWiseSum(3, 3)
        self.eval()

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor, offset: int):
        # rgb: CHW, alpha: CHW
        assert rgb.ndim == 3 and alpha.ndim == 3 and rgb.shape[0] == 3 and alpha.shape[0] == 1
        rgb = rgb.clone()
        alpha = alpha.squeeze(0)
        mask = alpha.new_zeros(alpha.shape)
        mask[alpha > 0] = 1.
        mask_nega = mask < 1.
        rgb[:, mask_nega] = 0.
        for i in range(offset):
            mask_weight = self.sum_alpha(mask)
            border = self.sum_rgb(rgb)
            border /= mask_weight + 1e-7
            rgb[:, mask_nega] = border[:, mask_nega]
            mask.zero_()
            mask[mask_weight > 0] = 1.
            mask_nega = mask < 1.

        return rgb.clamp_(0., 1.)
