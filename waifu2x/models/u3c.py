# u3c discriminator
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import register_model
from nunif.modules.attention import SEBlock
from nunif.modules.res_block import ResBlockSNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from nunif.modules.avg_pool2d import ConvAvgPool2d
from .disc_utils import (
    Discriminator,
    normalize,
    modcrop,
    fit_to_size,
    apply_patch_project,
    bench,
)


class SNImageToConditionPatch8(nn.Module):
    # 1/8 resolution patch
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            spectral_norm(nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate")),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockSNLReLU(embed_dim, embed_dim, stride=2),
            ResBlockSNLReLU(embed_dim, embed_dim),
            SEBlock(embed_dim, bias=True),
        )
        self.fc = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0)),
            )
            for out_channels in outputs
        ])
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x):
        x = normalize(x)
        x = self.features(x)
        outputs = []
        for fc in self.fc:
            enc = fc(x)
            outputs.append(enc)
        return outputs


class PoolFormerBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=7, conv_kernel_size=1, groups=4):
        super().__init__()
        self.mlp = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=conv_kernel_size, stride=1,
                                    padding=(conv_kernel_size - 1) // 2, bias=False)),
        )
        self.avg_pool2d = ConvAvgPool2d(in_channels, kernel_size=kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, count_include_pad=False)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.kernel_size = kernel_size

    def pooling(self, x):
        pool = self.avg_pool2d(x)
        x = pool - x
        return x

    def forward(self, x):
        x = x + self.pooling(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register_model
class U3ConditionalDiscriminator(Discriminator):
    name = "waifu2x.u3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.5, 0.5))
        C1 = 64
        C2 = 128
        C3 = 256
        C4 = 384
        pool_block = PoolFormerBlock
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, C1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C1, C2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(C2, C3, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(C3, C4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockSNLReLU(C4, C4),
        )
        self.enc1_proj = spectral_norm(nn.Conv2d(C2, C2, kernel_size=1, stride=1, padding=0))
        self.enc2_proj = spectral_norm(nn.Conv2d(C3, C3, kernel_size=1, stride=1, padding=0))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            spectral_norm(nn.Conv2d(C4, C3, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2 = nn.Sequential(
            ResBlockSNLReLU(C3, C3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            spectral_norm(nn.Conv2d(C3, C2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            pool_block(C2, kernel_size=7),
            pool_block(C2, kernel_size=7),
        )
        self.to_cond = SNImageToConditionPatch8(C1, [C2])
        self.request_false_condition = True
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 8)
        x = normalize(x)
        c = fit_to_size(x, c)
        cond = self.to_cond(c)[0]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.up1(x3) + self.enc2_proj(x2)
        x5 = self.up2(x4) + self.enc1_proj(x1)
        h = self.classifier(x5)
        assert h.shape[-1] / 4 == cond.shape[-1]
        z = apply_patch_project(h, cond)
        z = F.pad(z, (-8,) * 4)
        return z


if __name__ == "__main__":
    bench("waifu2x.u3_conditional_discriminator", compile=True)
    pass
