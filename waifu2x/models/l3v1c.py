# l3v1c discriminator
import random
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import register_model
from nunif.modules.attention import SEBlock
from nunif.modules.res_block import ResBlockGNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from .disc_utils import (
    Discriminator,
    normalize,
    modcrop,
    fit_to_size,
    bench
)


class ImageToCondition(nn.Module):
    # 1x1 vector
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


def group_norm(dim):
    return nn.GroupNorm(32, dim)


@register_model
class L3Discriminator(Discriminator):
    name = "waifu2x.l3_discriminator"

    def __init__(self, in_channels=3, out_channels=1, negative_slope=0.2):
        super().__init__(locals())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate")
        self.features = nn.Sequential(
            group_norm(64),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            group_norm(128),
            nn.LeakyReLU(negative_slope, inplace=True),
            SEBlock(128, bias=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            group_norm(256),
            nn.LeakyReLU(negative_slope, inplace=True),
            SEBlock(256, bias=True),
            ResBlockGNLReLU(256, 512),
            SEBlock(512, bias=True),
            spectral_norm(nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1)))
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        x = normalize(x)
        x = self.features(self.first_layer(x))
        x = self.classifier(x)
        x = F.pad(x, (-8,) * 4)
        return x


@register_model
class L3ConditionalDiscriminator(L3Discriminator):
    name = "waifu2x.l3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1, negative_slope=0.2):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         negative_slope=negative_slope)
        self.to_cond = ImageToCondition(32, [64, 256])

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        c = fit_to_size(x, c)
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        x = F.pad(x, (-8,) * 4)
        return x


class V1Discriminator(Discriminator):
    def __init__(self, in_channels=3, out_channels=1, negative_slope=0.2):
        super().__init__(locals())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.features = nn.Sequential(
            group_norm(64),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            group_norm(64),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.classifier = nn.Sequential(
            group_norm(128),
            nn.LeakyReLU(negative_slope, inplace=True),
            SEBlock(128, bias=True),
            spectral_norm(nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)),
        )
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        x = normalize(x)
        x = self.features(self.first_layer(x))
        x = self.classifier(x)
        x = F.pad(x, (-32,) * 4)
        return x


@register_model
class V1ConditionalDiscriminator(V1Discriminator):
    name = "waifu2x.v1_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1, negative_slope=0.2):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         negative_slope=negative_slope)
        self.to_cond = ImageToCondition(32, [64, 128])

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        c = fit_to_size(x, c)
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        x = F.pad(x, (-32,) * 4)
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
        return l3, v1


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
        return l3, v1


@register_model
class L3V1EnsembleConditionalDiscriminator(Discriminator):
    name = "waifu2x.l3v1_ensemble_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.8, 0.2))
        N = 3
        self.index = 0

        # additional variations. probably not very necessary.
        negative_slope = [0.2, 0.15, 0.1]

        self.l3 = nn.ModuleList([
            L3ConditionalDiscriminator(in_channels=in_channels, out_channels=out_channels,
                                       negative_slope=negative_slope[i])
            for i in range(N)
        ])
        self.v1 = nn.ModuleList([
            V1ConditionalDiscriminator(in_channels=in_channels, out_channels=out_channels,
                                       negative_slope=negative_slope[i])
            for i in range(N)
        ])

    def round(self):
        # called from the trainer at every iteration.
        self.index = random.randint(0, len(self.l3) - 1)

    def forward(self, x, c=None, scale_factor=None):
        l3 = self.l3[self.index](x, c, scale_factor)
        v1 = self.v1[self.index](x, c, scale_factor)
        return l3, v1


def _bench():
    bench("waifu2x.l3v1_conditional_discriminator", compile=True)
    bench("waifu2x.l3v1_ensemble_conditional_discriminator", compile=True)


if __name__ == "__main__":
    _bench()
    pass
