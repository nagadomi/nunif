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


@register_model
class L3Discriminator(Model):
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

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
class R3Discriminator(Model):
    # resnet type
    name = "waifu2x.r3_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockGNLReLU(64, 128, stride=2),
            SEBlock(128, bias=True),
            ResBlockGNLReLU(128, 256, stride=2),
            SEBlock(256, bias=True),
        )
        self.classifier = nn.Sequential(
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class R3ConditionalDiscriminator(R3Discriminator):
    name = "waifu2x.r3_conditional_discriminator"

    def __init__(self, in_channels=6, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, c, scale_factor):
        x = normalize(x)
        c = scale_c(x, c, scale_factor, mode="bilinear")
        c = normalize(c)
        x = torch.cat([x, c], dim=1)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class V3Discriminator(Model):
    # vgg type
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

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model
class S3Discriminator(Model):
    # swin_transformer type
    name = "waifu2x.s3_discriminator"
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.swin = SwinTransformer(
            num_classes=1,
            patch_size=[2, 2],
            embed_dim=64,
            depths=[2, 2, 2],
            num_heads=[8, 8, 8],
            window_size=[6, 6],
            stochastic_depth_prob=0.,
        )
        self.classifier = nn.Sequential(
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))

    def forward_features(self, x):
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = self.swin.permute(x)
        return x

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    l3 = L3Discriminator()
    l3c = L3Discriminator()
    r3 = R3Discriminator()
    r3c = R3ConditionalDiscriminator()
    v3 = V3Discriminator()
    s3 = S3Discriminator()
    x = torch.zeros((1, 3, 256, 256))
    print(l3(x).shape)
    print(l3c(x, x, 1).shape)
    print(r3(x).shape)
    print(r3c(x, x, 1).shape)
    print(v3(x, x, 1).shape)
    print(s3(x, x, 1).shape)
