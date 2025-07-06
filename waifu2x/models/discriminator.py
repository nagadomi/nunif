import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, register_model
from nunif.modules.attention import SEBlock, SNSEBlock
from nunif.modules.res_block import ResBlockGNLReLU, ResBlockSNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init, icnr_init
from dctorch.functional import dct2, idct2
from nunif.modules.permute import window_partition2d, window_reverse2d
from nunif.modules.color import rgb_to_ycbcr, rgb_to_yrgb
import os


# Compiling the discriminator is disabled due to VRAM memory leak.
# I guess it is because the discriminator is sometimes used and sometimes not used.
# os.environ["NUNIF_DISC_COMPILE"] = "1"


def normalize(x):
    return x * 2. - 1.


def clamp(x, min=-2., max=2., eps=0.01):
    c = torch.clamp(x, min, max)
    return c - (c.detach() - x) * eps


def modcrop(x, n):
    if x.shape[2] % n != 0:
        unpad = (n - x.shape[2] % n) // 2
        x = F.pad(x, (-unpad,) * 4)
    return x


def modpad(x, n):
    rem = n - input.shape[2] % n
    pad1 = rem // 2
    pad2 = rem - pad1
    x = F.pad(x, (pad1, pad2, pad1, pad2))
    return x


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
        self.aggregate = nn.Linear(embed_dim * 16, embed_dim, bias=True)
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, out_channels, bias=True))
            for out_channels in outputs])
        basic_module_init(self)

    @conditional_compile("NUNIF_DISC_COMPILE")
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


class ImageToConditionPatch(nn.Module):
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            spectral_norm(nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate")),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockSNLReLU(embed_dim, embed_dim),
            ResBlockSNLReLU(embed_dim, embed_dim),
        )
        self.conv = spectral_norm(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, padding_mode="replicate"))
        self.fc = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0)
            )
            for out_channels in outputs])
        basic_module_init(self)

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x):
        # expect (64 * 4 - offset * 2) or (112 * 4 - offset * 2)
        avg_6x6 = x.shape[2] > 64 * 4
        x = normalize(x)
        x = self.features(x)
        if avg_6x6:
            x = F.adaptive_avg_pool2d(x, (6, 6))
        else:
            x = F.adaptive_avg_pool2d(x, (3, 3))
        x = self.conv(x)
        outputs = []
        for fc in self.fc:
            enc = fc(x)
            outputs.append(enc)
        return outputs


class PatchToCondition(nn.Module):
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(384, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim, out_channels, kernel_size=3,
                          stride=1, padding=1, padding_mode="replicate"),
            ) for out_channels in outputs])

    def forward(self, x):
        outputs = []
        for proj in self.proj:
            enc = proj(x)
            outputs.append(enc)
        return outputs


def fit_to_size(x, cond):
    dh = cond.shape[2] - x.shape[2]
    dw = cond.shape[3] - x.shape[3]
    assert dh >= 0 and dw >= 0
    pad_h, pad_w = dh // 2, dw // 2
    if pad_h > 0 or pad_w > 0:
        cond = F.pad(cond, (-pad_w, -pad_w, -pad_h, -pad_h))
    return cond


def add_bias(x, cond):
    cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
    return x + cond


def apply_attention(x, cond):
    cond = torch.sigmoid(cond)
    cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
    return x * cond


def add_noise(x, strength=0.01):
    B, C, H, W = x.shape
    noise1x = torch.randn((B, C, H, W), dtype=x.dtype, device=x.device)
    noise2x = torch.randn((B, C, H // 2, W // 2), dtype=x.dtype, device=x.device)
    noise2x = F.interpolate(noise2x, size=(H, W), mode="nearest")
    noise4x = torch.randn((B, C, H // 4, W // 4), dtype=x.dtype, device=x.device)
    noise4x = F.interpolate(noise4x, size=(H, W), mode="nearest")

    noise = (noise1x + noise2x + noise4x) * (strength / 3.0)
    return x + noise


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights


@register_model
class L3Discriminator(Discriminator):
    name = "waifu2x.l3_discriminator"

    def __init__(self, in_channels=3, out_channels=1, feature_se_block=True):
        super().__init__(locals())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate")
        se_block = SEBlock(128, bias=True) if feature_se_block else nn.Identity()
        self.features = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            se_block,

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0))
        basic_module_init(self)

    @conditional_compile("NUNIF_DISC_COMPILE")
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

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None, scale_factor=None):
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        return x


class V1Discriminator(Discriminator):
    def __init__(self, in_channels=3, out_channels=1, se_block=True):
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
        se_block = SEBlock(128, bias=True) if se_block else nn.Identity()
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            se_block,
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0),
        )
        basic_module_init(self)

    @conditional_compile("NUNIF_DISC_COMPILE")
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

    @conditional_compile("NUNIF_DISC_COMPILE")
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
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            ResBlockSNLReLU(64, 128, stride=2),
            SNSEBlock(128, bias=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            ResBlockSNLReLU(128, 256, stride=2),
            SNSEBlock(256, bias=True))
        self.enc4 = ResBlockSNLReLU(256, 256, stride=1)
        self.class1 = nn.Sequential(
            ResBlockSNLReLU(256, 256, padding_mode="none"),
            SNSEBlock(256, bias=True),
            spectral_norm(nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=0))
        )
        self.up1 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 128 * 4, kernel_size=1, stride=1, padding=0)),
            nn.PixelShuffle(2))
        self.dec1 = ResBlockSNLReLU(128, 128)
        self.class2 = nn.Sequential(
            ResBlockSNLReLU(128, 128, padding_mode="none"),
            SNSEBlock(128, bias=True),
            spectral_norm(nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0))
        )
        self.up2 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 64 * 4, kernel_size=1, stride=1, padding=0)),
            nn.PixelShuffle(2))

        self.dec2 = ResBlockSNLReLU(64, 128, padding_mode="none")
        self.class3 = nn.Sequential(
            ResBlockSNLReLU(128, 128, padding_mode="none"),
            SNSEBlock(128, bias=True),
            spectral_norm(nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0)),
        )
        basic_module_init(self)
        icnr_init(self.up1[0], scale_factor=2)
        icnr_init(self.up2[0], scale_factor=2)
        self.to_cond = ImageToConditionPatch(64, [128])

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        cond = self.to_cond(fit_to_size(x, c))
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x2 = F.leaky_relu(add_bias(x2, cond[0]), 0.2, inplace=True)
        x3 = self.enc3(x2)
        x3 = self.enc4(x3)
        z1 = self.class1(x3)
        x4 = self.dec1(self.up1(x3) + x2)
        z2 = self.class2(x4)
        z3 = self.class3(self.dec2(self.up2(x4) + x1))

        return z1, z2, z3


@register_model
class DCTConditionalDiscriminator(Discriminator):
    name = "waifu2x.dct_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        dim = 256
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels * 8 * 8, dim, kernel_size=1, stride=1, padding=0)),
            ResBlockSNLReLU(dim, dim),
            ResBlockSNLReLU(dim, dim),
        )
        self.classifier = nn.Sequential(
            ResBlockSNLReLU(dim, dim),
            SNSEBlock(dim, bias=True),
            ResBlockSNLReLU(dim, dim),
            SNSEBlock(dim, bias=True),
            ResBlockSNLReLU(dim, dim),
            spectral_norm(nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=0)),
        )
        self.to_cond = ImageToConditionPatch(32, [dim])

    @staticmethod
    def window_dct(x, window_size, stride):
        B, C, H, W = x.shape
        x = F.unfold(x, kernel_size=window_size, stride=stride, padding=0)
        x = x.view(B, C, window_size, window_size, -1).permute(0, 4, 1, 2, 3)
        N = x.shape[1]
        x = x.reshape(B * N, C, window_size, window_size).contiguous()
        # use idct
        x = dct2(x)
        C2 = C * window_size * window_size
        x = x.view(B, N, C2).permute(0, 2, 1).view(B, C2, int(N ** 0.5), int(N ** 0.5)).contiguous()
        return x

    @conditional_compile("NUNIF_DISC_COMPILE")
    def _forward(self, x, dct, c=None, scale_factor=None):
        c = fit_to_size(x, c)
        cond = self.to_cond(c)
        x = self.features(dct)
        x = apply_attention(x, cond[0])
        z = self.classifier(x)
        return z

    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        dct = self.window_dct(x, window_size=8, stride=4)
        return self._forward(x, dct, c, scale_factor)


if __name__ == "__main__":
    l3 = L3Discriminator()
    l3c = L3ConditionalDiscriminator()
    l3v1 = L3V1Discriminator()
    l3v1c = L3V1ConditionalDiscriminator()
    u3c = U3ConditionalDiscriminator()
    dct = DCTConditionalDiscriminator()

    S = 64 * 4 - 38 * 2
    x = torch.zeros((1, 3, S, S))
    c = torch.zeros((1, 3, S, S))
    print(l3(x, c, 4).shape)
    print(l3c(x, c, 4).shape)
    print([z.shape for z in l3v1(x, c, 4)])
    print([z.shape for z in l3v1c(x, c, 4)])
    print([z.shape for z in u3c(x, c, 4)])
    print(dct(x, c, 4).shape)
