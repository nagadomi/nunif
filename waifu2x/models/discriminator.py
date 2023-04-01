import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, get_model_config, register_model
from .cunet import UNet1, UNet2, CUNet, UpCUNet
from .swin_unet import SwinTransformerBlocks
from nunif.modules import SEBlock
from nunif.modules.res_block import ResBlockGNLReLU
from torchvision.models import SwinTransformer


def normalize(x):
    return x * 2. - 1.


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


def init_moduels(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights

    def get_config(self):
        config = dict(super().get_config())
        config.update({"loss_weights": self.loss_weights})
        return config


@register_model
class L3Discriminator(Discriminator):
    name = "waifu2x.l3_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
        )
        self.classifier = nn.Sequential(
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))

        init_moduels(self)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class L3ConditionalDiscriminator(L3Discriminator):
    name = "waifu2x.l3_conditional_discriminator"

    def __init__(self, in_channels=6, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        c = scale_c(x, c, scale_factor, mode="bilinear")
        c = normalize(c)
        x = torch.cat([x, c], dim=1)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class V3Discriminator(Model):
    name = "waifu2x.v3_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))

        init_moduels(self)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class V3SpatialDiscriminator(Discriminator):
    name = "waifu2x.v3_spatial_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.1, 0.1))
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 64),
                nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                SEBlock(128, bias=True),
                nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 256),
                nn.LeakyReLU(0.2, inplace=True),
                SEBlock(256, bias=True),
                nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.LeakyReLU(0.2, inplace=True))
        ])
        def make_classifier(in_channels, res_block):
            if res_block:
                mid_channels = in_channels * 2
                modules = [
                    ResBlockGNLReLU(in_channels, mid_channels),
                    SEBlock(mid_channels, bias=True),
                ]
            else:
                mid_channels = in_channels
                modules = []
            modules += [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1,
                          padding=0),
            ]
            return nn.Sequential(*modules)
        self.classifiers = nn.ModuleList([
            make_classifier(64, False),
            make_classifier(128, False),
            make_classifier(256, True)])
        init_moduels(self)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x1 = self.features[0](x)
        x2 = self.features[1](x1)
        x3 = self.features[2](x2)
        z1 = self.classifiers[0](x1)
        z2 = self.classifiers[1](x2)
        z3 = self.classifiers[2](x3)

        return z3, z2, z1


def vgg1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(32, 64),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(32, 128),
        nn.LeakyReLU(0.2, inplace=True),

        SEBlock(128, bias=True),
        nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0),
    )


@register_model
class L3V1Discriminator(Discriminator):
    name = "waifu2x.l3v1_discriminator"
    def __init__(self, in_channels=3, out_channels=1, normalize_fix=False):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        self.l3 = L3Discriminator(in_channels=in_channels, out_channels=out_channels)
        self.v1 = vgg1(in_channels, out_channels)
        self.normalize_fix = normalize_fix
        init_moduels(self.v1)

    def forward(self, x, c=None, scale_factor=None):
        l3 = self.l3(x, c, scale_factor)
        if getattr(self, "normalize_fix", None):
            x = normalize(x)
        v1 = self.v1(x)
        return l3, v1


@register_model
class L3V1ConditionalDiscriminator(Discriminator):
    name = "waifu2x.l3v1_conditional_discriminator"
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        self.l3 = L3Discriminator(in_channels=in_channels, out_channels=out_channels)
        self.v1 = vgg1(in_channels, out_channels)
        init_moduels(self.v1)

    def forward(self, x, c=None, scale_factor=None):
        c = scale_c(x, c, scale_factor, mode="bilinear")
        x = torch.cat([x, c], dim=1)
        l3 = self.l3(x)
        v1 = self.v1(normalize(x))
        return l3, v1


if __name__ == "__main__":
    l3 = L3Discriminator()
    l3c = L3Discriminator()
    v3 = V3Discriminator()
    v3s = V3SpatialDiscriminator()
    l3v1 = L3V1Discriminator()
    x = torch.zeros((1, 3, 256, 256))
    print(l3(x).shape)
    print(l3c(x, x, 1).shape)
    print(v3(x, x, 1).shape)
    print(v3s(x, x, 1).shape)
    print(l3v1(x, x, 1).shape)
