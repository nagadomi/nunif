import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, get_model_config, register_model
from .cunet import UNet1, UNet2, CUNet, UpCUNet
from nunif.modules import SEBlock


def scale_c(x, c, scale_factor, mode="nearest"):
    if mode in {"bilinear", "bicubic", "linear", "trilinear"}:
        c = F.interpolate(c, scale_factor=scale_factor, mode=mode, align_corners=False)
    else:
        c = F.interpolate(c, scale_factor=scale_factor, mode=mode)
    offset = (c.shape[2] - x.shape[2]) // 2
    if offset > 0:
        c = F.pad(c, (-offset, -offset, -offset, -offset), mode="constant")
    assert c.shape[2] == x.shape[2] and c.shape[3] == x.shape[3]
    return c

def add_noise(x, strength=0.01):
    B, C, H, W = x.shape
    noise1x = torch.randn((B, C, H, W), dtype=x.dtype, device=x.device)
    noise2x = torch.randn((B, C, H // 2, W // 2), dtype=x.dtype, device=x.device)
    noise2x = F.interpolate(noise2x, size=(H, W), mode="nearest")
    noise4x = torch.randn((B, C, H // 4, W // 4), dtype=x.dtype, device=x.device)
    noise4x = F.interpolate(noise4x, size=(H, W), mode="nearest")

    noise = (noise1x + noise2x + noise4x) * (strength / 3.0)
    return x + noise


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
class L3Discriminator(Model):
    name = "waifu2x.l3_discriminator"

    def __init__(self, out_channels=1):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(512, bias=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(512, bias=True),

            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, c=None, scale_factor=None):
        x = (x - 0.5) * 2.
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    u1 = UNet1Discriminator.from_cunet(CUNet())
    u2 = UNet2Discriminator.from_cunet(UpCUNet())

    x = torch.zeros((1, 3, 128, 128))
    print(u1(x).shape)
    print(u2(x).shape)
