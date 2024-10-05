import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, register_model
from nunif.modules.attention import SEBlock, SNSEBlock
from nunif.modules.res_block import ResBlockGNLReLU, ResBlockSNLReLU
from nunif.modules.dinov2 import DINOEmbedding, dinov2_normalize, dinov2_pad, DINO_PATCH_SIZE
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init, icnr_init
from dctorch.functional import dct2
from nunif.modules.permute import window_partition2d, window_reverse2d


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

    @conditional_compile("NUNIF_TRAIN")
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
            nn.AvgPool2d((2, 2)),
            nn.PixelUnshuffle(2),
            nn.Conv2d(3 * 4, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockGNLReLU(embed_dim, embed_dim),
        )
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, padding_mode="replicate")
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0)
            )
            for out_channels in outputs])
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
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


class DINOPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = DINOEmbedding()

    def forward(self, x):
        B, C, H, W = x.shape
        x = dinov2_pad(x)
        x = dinov2_normalize(x)
        x = self.dino.forward_patch(x)
        x = x.detach()
        return x


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
    cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
    return x + cond


def dinov2_add_patch(x, cond):
    cond = F.leaky_relu(x, 0.1, inplace=False)
    cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
    return x + cond


def dinov2_attention_patch(x, cond):
    cond = torch.sigmoid(cond)
    cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
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

    @conditional_compile("NUNIF_TRAIN")
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

    @conditional_compile("NUNIF_TRAIN")
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

    @conditional_compile("NUNIF_TRAIN")
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

    @conditional_compile("NUNIF_TRAIN")
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


@register_model
class L3V1DINOConditionalDiscriminator(Discriminator):
    name = "waifu2x.l3v1_dino_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        self.l3 = L3Discriminator(in_channels=in_channels, out_channels=out_channels)
        self.v1 = V1Discriminator(in_channels=in_channels, out_channels=out_channels)
        self.dino_patch = DINOPatch()
        self.to_cond_l3 = PatchToCondition(32, [64, 256])
        self.to_cond_v1 = PatchToCondition(32, [64, 128])

    def forward_l3(self, x, cond):
        x = self.l3.first_layer(x)
        x = dinov2_attention_patch(x, cond[0])
        x = self.l3.features(x)
        x = dinov2_attention_patch(x, cond[1])
        x = self.l3.classifier(x)
        return x

    def forward_v1(self, x, cond):
        x = self.v1.first_layer(x)
        x = dinov2_attention_patch(x, cond[0])
        x = self.v1.features(x)
        x = dinov2_attention_patch(x, cond[1])
        x = self.v1.classifier(x)
        return x

    def forward(self, x, c=None, scale_factor=None):
        dino_patch = self.dino_patch(c)
        cond_v1 = self.to_cond_v1(dino_patch)
        cond_l3 = self.to_cond_l3(dino_patch)
        x = normalize(x)
        v1 = self.forward_v1(x, cond_v1)
        l3 = self.forward_l3(x, cond_l3)
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
        basic_module_init(self.enc1)
        icnr_init(self.up1[0], scale_factor=2)
        icnr_init(self.up2[0], scale_factor=2)
        self.to_cond = ImageToConditionPatch(64, [128])

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        cond = self.to_cond(fit_to_size(x, c))
        x = normalize(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x2 = F.leaky_relu(add_bias(x2, cond[0]), 0.2, inplace=True)
        x3 = self.enc3(x2)
        x3 = self.enc4(x3)
        z1 = self.class1(x3)
        x4 = self.dec1(self.up1(x3) + x2)
        z2 = self.class2(x4)
        z3 = self.class3(self.dec2(self.up2(x4) + x1))

        return clamp(z1), clamp(z2), clamp(z3)


@register_model
class DCTDiscriminator(Discriminator):
    name = "waifu2x.dct_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(1,))
        dim = 256
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels * 8 * 8, dim, kernel_size=1, stride=1, padding=0)),
            ResBlockSNLReLU(dim, dim),
            SNSEBlock(dim, bias=True),
            ResBlockSNLReLU(dim, dim),
        )
        self.classifier = nn.Sequential(
            ResBlockSNLReLU(dim, dim),
            SNSEBlock(dim, bias=True),
            ResBlockSNLReLU(dim, dim),
            spectral_norm(nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=0)),
        )
        self.to_cond = ImageToConditionPatch(64, [dim])

    @staticmethod
    def window_dct(x, window_size):
        z = window_partition2d(x, window_size=window_size)
        B, N, C, H, W = z.shape
        z = z.reshape(B * N, C, H, W)
        z = dct2(z)
        z = z.reshape(B, N, C, H, W)
        z = window_reverse2d(z, x.shape, window_size=window_size)
        z = F.pixel_unshuffle(z, window_size)
        return z.contiguous()

    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        c = fit_to_size(x, c)
        cond = self.to_cond(c)
        x = self.window_dct(x, window_size=8)
        x = self.features(x)
        x = add_bias(x, cond[0])
        z = self.classifier(x)

        return z


if __name__ == "__main__":
    l3 = L3Discriminator()
    l3c = L3ConditionalDiscriminator()
    l3v1 = L3V1Discriminator()
    l3v1c = L3V1ConditionalDiscriminator()
    l3v1dino = L3V1DINOConditionalDiscriminator()
    u3c = U3ConditionalDiscriminator()
    dct = DCTDiscriminator()

    S = 64 * 4 - 38 * 2
    x = torch.zeros((1, 3, S, S))
    c = torch.zeros((1, 3, S, S))
    print(l3(x, c, 4).shape)
    print(l3c(x, c, 4).shape)
    print([z.shape for z in l3v1(x, c, 4)])
    print([z.shape for z in l3v1c(x, c, 4)])
    print([z.shape for z in l3v1dino(x, c, 4)])
    print([z.shape for z in u3c(x, c, 4)])
    print(dct(x, c, 4).shape)
