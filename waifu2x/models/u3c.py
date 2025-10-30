# u3c discriminator for art_scan
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import register_model
from nunif.modules.pad import get_pad_size
from nunif.modules.replication_pad2d import ReplicationPad2dNaive
from nunif.modules.attention import SEBlock, WindowCrossMHA2d, WindowScoreBias
from nunif.modules.res_block import ResBlockSNLReLU, ResBlockGNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init, icnr_init
from nunif.modules.avg_pool2d import ConvAvgPool2d
from .disc_utils import (
    Discriminator,
    normalize,
    modpad,
    fit_to_size,
    to_y,
    bench,
)


class ImageToConditionPatch8(nn.Module):
    # 1/8 resolution patch
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockGNLReLU(embed_dim, embed_dim, stride=2, bias=False, padding_mode="replicate"),
            SEBlock(embed_dim, bias=True),
            ResBlockGNLReLU(embed_dim, embed_dim, bias=False, padding_mode="replicate"),
        )
        self.fc = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0)),
            )
            for out_channels in outputs
        ])
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x):
        x = normalize(x)
        x = F.interpolate(x, size=(x.shape[-2] // 4, x.shape[-1] // 4),
                          mode="bilinear", antialias=True, align_corners=False)
        x = self.features(x)
        outputs = []
        for fc in self.fc:
            enc = fc(x)
            outputs.append(enc)
        return outputs


class ImageToCondition(nn.Module):
    # 1x1 vector
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            nn.Conv2d(4, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, embed_dim),
            nn.ReLU(inplace=True),
            ResBlockGNLReLU(embed_dim, embed_dim, stride=2, bias=False),
            SEBlock(embed_dim, bias=True),
            ResBlockGNLReLU(embed_dim, embed_dim, bias=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.aggregate = spectral_norm(nn.Linear(embed_dim * 16, embed_dim, bias=False))
        self.fc = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Linear(embed_dim, embed_dim, bias=False)),
                nn.ReLU(inplace=True),
                spectral_norm(nn.Linear(embed_dim, out_channels, bias=False))
            )
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


class PoolFormerBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=7, conv_kernel_size=1, groups=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=conv_kernel_size, stride=1,
                      padding=(conv_kernel_size - 1) // 2, bias=False),
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


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        num_heads = max(in_channels // 32, 1)
        self.window_size = 4
        self.mha = WindowCrossMHA2d(in_channels, num_heads=num_heads, window_size=self.window_size)
        self.bias = WindowScoreBias(window_size=self.window_size)
        self.norm1 = nn.LayerNorm(in_channels, bias=False)
        self.norm2 = nn.LayerNorm(in_channels, bias=False)

    def forward(self, x, cond):
        # Note: query=cond, kv=x, x = x + kv(x)
        pad = get_pad_size(x, self.window_size)
        x = F.pad(x, pad, mode="constant", value=0)
        cond = F.pad(cond, pad, mode="constant", value=0)
        out = self.mha(cond, x, attn_mask=self.bias(), layer_norm1=self.norm1, layer_norm2=self.norm2)
        x = x + out
        x = F.pad(x, [-p for p in pad])
        return x


@register_model
class U3ConditionalDiscriminator(Discriminator):
    name = "waifu2x.u3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals(), loss_weights=(0.9, 0.1))
        C1 = 32
        C2 = 64
        C3 = 128
        C4 = 256
        padding_mode = "replicate"

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, C1, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C1, C2, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1_proj = spectral_norm(nn.Conv2d(C2, C2, kernel_size=1, stride=1, padding=0, bias=False))
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(C2, C3, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2_proj = spectral_norm(nn.Conv2d(C3, C3, kernel_size=1, stride=1, padding=0, bias=False))
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(C3, C4, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            ResBlockSNLReLU(C4, C4, padding_mode=padding_mode, bias=False),
            ResBlockSNLReLU(C4, C4, padding_mode=padding_mode, bias=False),
        )
        self.class1 = nn.Sequential(
            ResBlockSNLReLU(C4, C4, padding_mode=padding_mode),
            spectral_norm(nn.Conv2d(C4, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)),
        )
        self.up1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(C4, C3, kernel_size=2, stride=2, padding=0, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            ResBlockSNLReLU(C3, C3, padding_mode=padding_mode, bias=False),
        )
        self.up2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(C3, C2, kernel_size=2, stride=2, padding=0, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            ResBlockSNLReLU(C2, C2, padding_mode=padding_mode, bias=False),
        )
        self.class2 = nn.Sequential(
            ResBlockSNLReLU(C2, C2, padding_mode=padding_mode),
            spectral_norm(nn.Conv2d(C2, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)),
        )
        basic_module_init(self)
        self.to_cond = ImageToConditionPatch8(64, [C4])
        #self.cross_attention = CrossAttention(C4)
        self.request_false_condition = True

    def _debug_save_cond(self, x, c):
        import os
        import torchvision.transforms.functional as TF
        save_dir = "./tmp/_gan_cond"
        os.makedirs(save_dir, exist_ok=True)
        self._save_count = getattr(self, "_save_count", 0) + 1
        for i in range(x.shape[0]):
            TF.to_pil_image(x[i].clamp(0, 1)).save(os.path.join(save_dir, f"{i + self._save_count}_x.png"))
            TF.to_pil_image(c[i].clamp(0, 1)).save(os.path.join(save_dir, f"{i + self._save_count}_c.png"))

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None, scale_factor=None):
        x = torch.cat([x, to_y(x)], dim=1)
        c = torch.cat([c, to_y(c)], dim=1)
        c = fit_to_size(x, c)
        x = modpad(x, 16)
        c = modpad(c, 16)
        # self._debug_save_cond(x, c)
        cond = self.to_cond(c)
        x = normalize(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x3 = x3 + cond[0]
        # x3 = self.cross_attention(x3, cond[0])
        x3 = self.enc4(x3)
        z1 = self.class1(x3)
        x4 = self.dec1(self.up1(x3) + self.enc2_proj(x2))
        x5 = self.dec2(self.up2(x4) + self.enc1_proj(x1))
        z2 = self.class2(x5)

        if self.training:
            return F.pad(z2, (-8,) * 4), F.pad(z1, (-2,) * 4)
        else:
            return z2, z1


@register_model
class U3CEnsembleConditionalDiscriminator(Discriminator):
    name = "waifu2x.u3_ensemble_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1, imbalanced_prob=False):
        super().__init__(locals())
        N = 3
        if imbalanced_prob:
            self.prob = [1.0, 0.5, 0.25]
        else:
            self.prob = [1.0, 1.0, 1.0]
        self.prob = [p / sum(self.prob) for p in self.prob]
        self.indexes = [0, 1, 2]
        self.index = 0

        self.u3c = nn.ModuleList([
            U3ConditionalDiscriminator(in_channels=in_channels, out_channels=out_channels)
            for i in range(N)
        ])

    def round(self):
        # called from the trainer at every iteration.
        self.index = random.choices(self.indexes, weights=self.prob, k=1)[0]

    def forward(self, x, c=None, scale_factor=None):
        return self.u3c[self.index](x, c, scale_factor)


if __name__ == "__main__":
    bench("waifu2x.u3_conditional_discriminator", compile=False)
    bench("waifu2x.u3_ensemble_conditional_discriminator", compile=True)
    pass
