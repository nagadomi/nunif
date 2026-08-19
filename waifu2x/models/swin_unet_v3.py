import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.models import I2IBaseModel, register_model
from nunif.modules.attention import WindowMHA2dV2
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.glu import align_up, swiglu
from nunif.modules.init import basic_module_init
from nunif.modules.norm import RMSNorm1, RMSNorm2d
from nunif.modules.replication_pad2d import replication_pad2d_auto
from nunif.modules.rope import RoPE2d


class SwiGLUConv2d(nn.Module):
    def __init__(self, in_channels, mlp_ratio=3, padding=True):
        super().__init__()
        mid = align_up(in_channels, mlp_ratio * 0.5, 32)
        self.norm = RMSNorm2d(in_channels)
        self.w1 = nn.Conv2d(in_channels, mid * 2, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid, in_channels, kernel_size=3, stride=1, padding=0)
        self.padding = padding
        basic_module_init(self)

    def forward(self, x):
        x = self.norm(x)
        x = self.w1(x)
        x = swiglu(x, dim=1)
        if self.padding:
            x = replication_pad2d_auto(x, (1,) * 4, self.training)
        x = self.w2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio=3):
        super(ResBlock, self).__init__()
        self.conv = SwiGLUConv2d(in_channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class ResBlocks(nn.Module):
    def __init__(self, in_channels, num_layers, mlp_ratio=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x


class WACBlock(nn.Module):
    def __init__(self, in_channels, num_heads, window_size, mlp_ratio=3, shift=False):
        super(WACBlock, self).__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.norm1 = RMSNorm1(in_channels)
        self.rope = RoPE2d(in_channels // num_heads, size=window_size, norm_layer=RMSNorm1)
        self.mha = WindowMHA2dV2(in_channels, num_heads, window_size=window_size, shift=shift)
        self.mlp = SwiGLUConv2d(in_channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.mha(x, layer_norm=self.norm1, rope=self.rope)
        x = x + self.mlp(x)

        return x


class WACBlocks(nn.Module):
    def __init__(self, in_channels, num_heads, window_size, num_layers, mlp_ratio=3, shift=None):
        assert num_layers % 2 == 0
        super(WACBlocks, self).__init__()
        if isinstance(window_size, int):
            window_size = [window_size] * num_layers
        if shift is None:
            # True, False, ..., False
            shift = [i % 2 == 0 for i in range(num_layers)]
            assert num_layers % 2 == 0

        self.blocks = nn.ModuleList(
            [
                WACBlock(
                    in_channels,
                    window_size=window_size[i],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    shift=shift[i],
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        basic_module_init(self)

    def forward(self, x):
        x = replication_pad2d_auto(x, (1,) * 4, self.training)
        x = self.conv(x)
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        basic_module_init(self)

    def forward(self, x, shortcut):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, shortcut), dim=1)
        x = replication_pad2d_auto(x, (1,) * 4, self.training)
        x = self.fuse(x)

        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.scale_bias = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.proj = nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=1, stride=1, padding=0)
        basic_module_init(self.proj)

    def forward(self, x, src):
        if self.scale_factor == 1:
            x = self.proj(x)
            x = src + x * self.scale_bias
        elif self.scale_factor == 2:
            x = self.proj(x)
            x = F.pixel_shuffle(x, 2)
            src = F.interpolate(src, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            x = src + x * self.scale_bias
        elif self.scale_factor == 4:
            x = self.proj(x)
            x = F.pixel_shuffle(x, 4)
            src = F.interpolate(src, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            x = src + x * self.scale_bias

        return x


def compute_head_dim(n):
    assert n % 32 == 0
    return n // 32


class SwinUNetV3Base(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        encoder_dim=64,
        middle_dim=128,
        decoder_dim=64,
        scale_factor=2,
    ):
        super(SwinUNetV3Base, self).__init__()
        assert scale_factor in {1, 2, 4}
        self.scale_factor = scale_factor
        C1 = encoder_dim
        C2 = middle_dim
        C3 = decoder_dim
        HEAD2 = compute_head_dim(C2)
        self.emb = nn.Sequential(
            nn.Conv2d(in_channels, C1, kernel_size=3, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=0),
        )
        self.res1_1 = ResBlocks(C1, num_layers=2)
        self.down1 = PatchDown(C1, C2)
        self.wac2 = WACBlocks(C2, window_size=8, num_heads=HEAD2, num_layers=8)
        self.up1 = PatchUp(C1 + C2, C3)
        self.res1_2 = ResBlocks(C3, num_layers=3)
        self.to_image = ToImage(C3, out_channels, scale_factor=scale_factor)

        basic_module_init(self.emb)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        src = x
        x = replication_pad2d_auto(x, (2,) * 4, self.training)
        x = self.emb(x)
        x1 = self.res1_1(x)
        x = self.down1(x1)
        x = self.wac2(x)
        x = self.up1(x, x1)
        x = self.res1_2(x)
        x = self.to_image(x, src)

        x = F.pad(x, (-8 * self.scale_factor,) * 4)
        if self.training:
            return x
        else:
            return x


def tile_size_validator(size):
    return size >= 64 and size % 32 == 0


@register_model
class SwinUNet1xV3(I2IBaseModel):
    name = "waifu2x.swin_unet_v3_1x"

    def __init__(self, in_channels=3, out_channels=3, encoder_dim=64, middle_dim=128, decoder_dim=64, **kwargs):
        super(SwinUNet1xV3, self).__init__(locals(), scale=1, offset=8, in_channels=in_channels, blend_size=4)
        self.register_tile_size_validator(tile_size_validator)
        self.unet = SwinUNetV3Base(
            in_channels,
            out_channels,
            encoder_dim=encoder_dim,
            middle_dim=middle_dim,
            decoder_dim=decoder_dim,
            scale_factor=1,
        )

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return z.clamp(0, 1)


@register_model
class SwinUNet2xV3(I2IBaseModel):
    name = "waifu2x.swin_unet_v3_2x"

    def __init__(self, in_channels=3, out_channels=3, encoder_dim=64, middle_dim=128, decoder_dim=96, **kwargs):
        super(SwinUNet2xV3, self).__init__(locals(), scale=2, offset=16, in_channels=in_channels, blend_size=8)
        self.register_tile_size_validator(tile_size_validator)
        self.unet = SwinUNetV3Base(
            in_channels,
            out_channels,
            encoder_dim=encoder_dim,
            middle_dim=middle_dim,
            decoder_dim=decoder_dim,
            scale_factor=2,
        )

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return z.clamp(0.0, 1.0)


@register_model
class SwinUNet4xV3(I2IBaseModel):
    name = "waifu2x.swin_unet_v3_4x"

    def __init__(self, in_channels=3, out_channels=3, encoder_dim=64, middle_dim=192, decoder_dim=128, **kwargs):
        super(SwinUNet4xV3, self).__init__(locals(), scale=4, offset=32, in_channels=in_channels, blend_size=16)
        self.register_tile_size_validator(tile_size_validator)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = SwinUNetV3Base(
            in_channels,
            out_channels,
            encoder_dim=encoder_dim,
            middle_dim=middle_dim,
            decoder_dim=decoder_dim,
            scale_factor=4,
        )

    def forward(self, x):
        z = self.unet(x)

        if self.training:
            return z
        else:
            return torch.clamp(z, 0.0, 1.0)


def _bench(name, compile):
    import time

    from nunif.models import create_model

    N = 20
    B = 4
    S = (256, 256)
    device = "cuda:0"

    model = create_model(name, in_channels=3, out_channels=3).to(device).eval()
    if compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x)
        print(z.shape)
        param = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{param:,}", f"compile={compile}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    torch.cuda.synchronize()
    et = time.time() - t
    print(et, 1 / (et / (B * N)), "FPS")


if __name__ == "__main__":
    enable_full_compile = True
    _bench("waifu2x.swin_unet_v3_1x", enable_full_compile)
    _bench("waifu2x.swin_unet_v3_2x", enable_full_compile)
    _bench("waifu2x.swin_unet_v3_4x", enable_full_compile)
