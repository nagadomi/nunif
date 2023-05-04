import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, register_model
from nunif.modules.attention import SEBlock, SNSEBlock
from nunif.modules.res_block import ResBlockGNLReLU, ResBlockSNLReLU
from nunif.modules.fourier_unit import FourierUnitSNLReLU
from torch.nn.utils.parametrizations import spectral_norm


def normalize(x):
    return x * 2. - 1.


def clamp(x, min=-2., max=2., eps=0.01):
    c = torch.clamp(x, min, max)
    return c + (c.detach() - x) * eps


class ImageToCondition(nn.Module):
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.aggregate = nn.LazyLinear(embed_dim, bias=True)
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, out_channels, bias=True))
            for out_channels in outputs])

    def forward(self, x):
        B = x.shape[0]
        x = normalize(x)
        x = self.features(x)
        x = self.aggregate(x.view(B, -1))
        outputs = []
        for fc in self.fc:
            enc = fc(x)
            enc = enc.view(B, enc.shape[1], 1, 1)
            outputs.append(enc)
        return outputs


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
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate")
        self.features = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))
        init_moduels(self)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(self.first_layer(x))
        x = self.classifier(x)
        return x


@register_model
class L3ConditionalDiscriminator(L3Discriminator):
    name = "waifu2x.l3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.to_cond = ImageToCondition(32, [64, 256])

    def forward(self, x, c=None, scale_factor=None):
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        return x


class V1Discriminator(Discriminator):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.features = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0),
        )
        init_moduels(self)

    def forward(self, x, c=None, scale_factor=None):
        x = normalize(x)
        x = self.features(self.first_layer(x))
        x = self.classifier(x)
        return x


@register_model
class V1ConditionalDiscriminator(V1Discriminator):
    name = "waifu2x.v1_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.to_cond = ImageToCondition(32, [64, 128])

    def forward(self, x, c=None, scale_factor=None):
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        return x


@register_model
class L3V1Discriminator(Discriminator):
    name = "waifu2x.l3v1_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        self.l3 = L3Discriminator(in_channels=in_channels, out_channels=out_channels)
        self.v1 = V1Discriminator(in_channels=in_channels, out_channels=out_channels)
        init_moduels(self.v1)

    def forward(self, x, c=None, scale_factor=None):
        l3 = self.l3(x, c, scale_factor)
        v1 = self.v1(x, c, scale_factor)
        return clamp(l3), clamp(v1)


@register_model
class L3V1ConditionalDiscriminator(Discriminator):
    name = "waifu2x.l3v1_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        self.l3 = L3ConditionalDiscriminator(in_channels=in_channels, out_channels=out_channels)
        self.v1 = V1ConditionalDiscriminator(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, c=None, scale_factor=None):
        l3 = self.l3(x, c, scale_factor)
        v1 = self.v1(x, c, scale_factor)
        return clamp(l3), clamp(v1)


class SelfSupervisedDiscriminator():
    pass


@register_model
class U3ConditionalDiscriminator(Discriminator):
    name = "waifu2x.u3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.333, 0.333, 0.333))
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,
                      padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            ResBlockSNLReLU(64, 128, stride=2),
            SNSEBlock(128, bias=True))
        self.enc3 = nn.Sequential(
            ResBlockSNLReLU(128, 256, stride=2),
            SNSEBlock(256, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.class1 = nn.Sequential(
            ResBlockSNLReLU(256, 256),
            SNSEBlock(256, bias=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=0)
        )
        self.up1 = spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0))
        self.dec1 = ResBlockSNLReLU(128, 128)
        self.class2 = nn.Sequential(
            ResBlockSNLReLU(128, 128),
            SNSEBlock(128, bias=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0)
        )
        self.up2 = spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0))
        self.dec2 = ResBlockSNLReLU(64, 64)
        self.class3 = nn.Sequential(
            ResBlockSNLReLU(64, 64),
            SNSEBlock(64, bias=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=0)
        )
        init_moduels(self)
        self.to_cond = ImageToCondition(64, [256])

    def forward(self, x, c=None, scale_factor=None):
        cond = self.to_cond(c)
        x = normalize(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = F.leaky_relu(self.enc3(x2) + cond[0], 0.2, inplace=True)
        z1 = self.class1(x3)

        x4 = self.dec1(self.up1(x3) + x2)
        z2 = self.class2(x4)
        z3 = self.class3(self.dec2(self.up2(x4) + x1))

        return clamp(z1), clamp(z2), clamp(z3)


@register_model
class U3FFTConditionalDiscriminator(Discriminator):
    name = "waifu2x.u3fft_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.333, 0.333, 0.333))
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,
                      padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            ResBlockSNLReLU(64, 128, stride=2),
            FourierUnitSNLReLU(128, 128),
            SNSEBlock(128, bias=True))
        self.enc3 = nn.Sequential(
            ResBlockSNLReLU(128, 256, stride=2),
            FourierUnitSNLReLU(256, 256),
            SNSEBlock(256, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.class1 = nn.Sequential(
            ResBlockSNLReLU(256, 256),
            SNSEBlock(256, bias=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=0)
        )
        self.up1 = spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0))
        self.dec1 = ResBlockSNLReLU(128, 128)
        self.class2 = nn.Sequential(
            ResBlockSNLReLU(128, 128),
            SNSEBlock(128, bias=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0)
        )
        self.up2 = spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0))
        self.dec2 = ResBlockSNLReLU(64, 64)
        self.class3 = nn.Sequential(
            ResBlockSNLReLU(64, 64),
            SNSEBlock(64, bias=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=0)
        )
        init_moduels(self)
        self.to_cond = ImageToCondition(64, [256])

    def forward(self, x, c=None, scale_factor=None):
        cond = self.to_cond(c)
        x = normalize(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = F.leaky_relu(self.enc3(x2) + cond[0], 0.2, inplace=True)
        z1 = self.class1(x3)

        x4 = self.dec1(self.up1(x3) + x2)
        z2 = self.class2(x4)
        z3 = self.class3(self.dec2(self.up2(x4) + x1))

        return clamp(z1), clamp(z2), clamp(z3)


if __name__ == "__main__":
    l3 = L3Discriminator()
    l3c = L3ConditionalDiscriminator()
    l3v1 = L3V1Discriminator()
    l3v1c = L3V1ConditionalDiscriminator()
    u3c = U3ConditionalDiscriminator()
    u3fftc = U3FFTConditionalDiscriminator()

    x = torch.zeros((1, 3, 192, 192))
    c = torch.zeros((1, 3, 192, 192))
    print(l3(x, c, 4).shape)
    print(l3c(x, c, 4).shape)
    print([z.shape for z in l3v1(x, c, 4)])
    print([z.shape for z in l3v1c(x, c, 4)])
    print([z.shape for z in u3c(x, c, 4)])
    print([z.shape for z in u3fftc(x, c, 4)])
