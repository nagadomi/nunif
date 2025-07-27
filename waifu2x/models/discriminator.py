import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import Model, register_model
from nunif.modules.attention import SEBlock, SNSEBlock
from nunif.modules.res_block import ResBlockGNLReLU, ResBlockSNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init


def normalize(x):
    return x * 2. - 1.


def modcrop(x, n):
    if x.shape[2] % n != 0:
        unpad = x.shape[2] % n // 2
        x = F.pad(x, (-unpad,) * 4)
    return x


def modpad(x, n):
    rem = n - input.shape[2] % n
    pad1 = rem // 2
    pad2 = rem - pad1
    x = F.pad(x, (pad1, pad2, pad1, pad2))
    return x


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


class ImageToConditionPatch8(nn.Module):
    # 1/8 resolution patch
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            spectral_norm(nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate")),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockSNLReLU(embed_dim, embed_dim, stride=2),
            ResBlockSNLReLU(embed_dim, embed_dim),
            SNSEBlock(embed_dim, bias=True),
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
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.kernel_size = kernel_size

    def pooling(self, x):
        pool = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1,
                            padding=(self.kernel_size - 1) // 2, count_include_pad=False)
        x = pool - x
        return x

    def forward(self, x):
        x = x + self.pooling(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def fit_to_size(x, cond):
    dh = cond.shape[2] - x.shape[2]
    dw = cond.shape[3] - x.shape[3]
    assert dh >= 0 and dw >= 0
    pad_h, pad_w = dh // 2, dw // 2
    if pad_h > 0 or pad_w > 0:
        cond = F.pad(cond, (-pad_w, -pad_w, -pad_h, -pad_h))
    return cond


def fit_to_size_x(x, cond, scale_factor):
    dh = cond.shape[2] - x.shape[2] // scale_factor
    dw = cond.shape[3] - x.shape[3] // scale_factor
    assert dh >= 0 and dw >= 0
    pad_h, pad_w = dh // 2, dw // 2
    if pad_h > 0 or pad_w > 0:
        cond = F.pad(cond, (-pad_w, -pad_w, -pad_h, -pad_h))
    return cond


def add_bias(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x + cond


def apply_scale(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cond = torch.sigmoid(cond)
    return x * cond


def apply_scale_bias(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    bias, scale = cond.chunk(2, dim=1)
    scale = torch.sigmoid(scale)
    return x * scale + bias


def apply_patch_project(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (x * cond).sum(dim=1, keepdim=True) * (x.shape[1] ** 0.5)


def apply_project(x, cond):
    if cond.shape[-1] != 1:
        cond = F.adaptive_avgpool2d(cond, (1, 1))
    return (x * cond).sum(dim=1, keepdim=True) * (x.shape[1] ** 0.5)


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
            spectral_norm(nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0)))
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

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
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
            spectral_norm(nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=0)),
        )
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 4)
        x = normalize(x)
        x = self.features(self.first_layer(x))
        x = self.classifier(x)
        x = F.pad(x, (-8,) * 4)
        return x


@register_model
class V1ConditionalDiscriminator(V1Discriminator):
    name = "waifu2x.v1_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.to_cond = ImageToCondition(32, [64, 128])

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x, c=None, scale_factor=None):
        x = modcrop(x, 4)
        c = fit_to_size(x, c)
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(self.first_layer(x) + cond[0])
        x = self.classifier(x + cond[1])
        x = F.pad(x, (-8,) * 4)
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
        self.to_cond = ImageToConditionPatch8(C1, [C2])
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


def _test():
    l3 = L3Discriminator()
    l3c = L3ConditionalDiscriminator()
    l3v1 = L3V1Discriminator()
    l3v1c = L3V1ConditionalDiscriminator()
    u3c = U3ConditionalDiscriminator()

    S = 64 * 4 - 38 * 2
    x = torch.zeros((1, 3, S, S))
    c = torch.zeros((1, 3, S, S))
    print(l3(x, c, 4).shape)
    print(l3c(x, c, 4).shape)
    print([z.shape for z in l3v1(x, c, 4)])
    print([z.shape for z in l3v1c(x, c, 4)])
    print([z.shape for z in u3c(x, c, 4)])


def _bench(name, compile=False):
    from nunif.models import create_model
    import time

    N = 20
    B = 4
    S = (256, 256)
    device = "cuda:0"

    model = create_model(name).to(device).eval()
    if compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    c = torch.zeros((B, 3, *S)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x, c, scale_factor=4)
        print(z.shape)
        param = sum([p.numel() for p in model.parameters()])
        print(model.name, f"{param:,}", f"compile={compile}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z, *_ = model(x, c, scale_factor=4)
    torch.cuda.synchronize()
    et = time.time() - t
    print(et, 1 / (et / (B * N)), "FPS")


if __name__ == "__main__":
    _test()
    # _bench("waifu2x.l3v1_conditional_discriminator", compile=True)
    _bench("waifu2x.u3_conditional_discriminator", compile=True)

    pass
