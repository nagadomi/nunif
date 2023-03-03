import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, get_model_config, register_model
from .cunet import UNet1, UNet2, CUNet, UpCUNet


@register_model
class UNet2Discriminator(Model):
    name = "waifu2x.unet2_discriminator"

    def __init__(self, in_channels=3):
        super().__init__(locals())
        self.unet = UNet2(in_channels=in_channels, out_channels=1, deconv=False)

    @staticmethod
    def from_cunet(cunet):
        assert isinstance(cunet, (CUNet, UpCUNet))
        discriminator = UNet2Discriminator(get_model_config(cunet, "i2i_in_channels"))
        discriminator.unet = cunet.unet2

        # replace last layer
        discriminator.unet.conv_bottom = nn.Conv2d(64, 1, 3, 1, 0)
        nn.init.kaiming_normal_(discriminator.unet.conv_bottom.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(discriminator.unet.conv_bottom.bias, 0)

        return discriminator

    def forward(self, x, c=None, scale_factor=None):
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0
        x = (x - 0.5) * 2.
        return self.unet(x)


@register_model
class UNet1Discriminator(Model):
    name = "waifu2x.unet1_discriminator"

    def __init__(self, in_channels=3):
        super().__init__(locals())
        self.unet = UNet1(in_channels=in_channels, out_channels=1, deconv=False)

    @staticmethod
    def from_cunet(cunet):
        assert isinstance(cunet, CUNet)
        discriminator = UNet1Discriminator(get_model_config(cunet, "i2i_in_channels"))
        discriminator.unet = cunet.unet1
        discriminator.unet.conv_bottom = nn.Conv2d(64, 1, 3, 1, 0)
        nn.init.kaiming_normal_(discriminator.unet.conv_bottom.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(discriminator.unet.conv_bottom.bias, 0)

        return discriminator

    def forward(self, x, c=None, scale_factor=None):
        assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0
        x = (x - 0.5) * 2.
        return self.unet(x)


@register_model
class UNet2ConditionalDiscriminator(Model):
    name = "waifu2x.unet2_cond_discriminator"

    def __init__(self, in_channels=3):
        super().__init__(locals())
        self.unet = UNet2(in_channels=in_channels*2, out_channels=1, deconv=False)

    def forward(self, x, c, scale_factor):
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0
        c = F.interpolate(c, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        offset = (c.shape[2] - x.shape[2]) // 2
        if offset > 0:
            c = F.pad(c, (-offset, -offset, -offset, -offset), mode="constant")
        assert c.shape[2] == x.shape[2] and c.shape[3] == x.shape[3]

        x = (x - 0.5) * 2.
        c = (c - 0.5) * 2.
        x = torch.cat([x, c], dim=1)
        return self.unet(x)


if __name__ == "__main__":
    u1 = UNet1Discriminator.from_cunet(CUNet())
    u2 = UNet2Discriminator.from_cunet(UpCUNet())

    x = torch.zeros((1, 3, 128, 128))
    print(u1(x).shape)
    print(u2(x).shape)
