import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model, register_model_factory
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.norm import RMSNorm1
from nunif.modules.replication_pad2d import ReplicationPad2dNaive as ReplicationPad2dDetach
from nunif.modules.init import icnr_init, basic_module_init


class GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=2, layer_norm=True):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        if layer_norm:
            self.norm = RMSNorm1((1, in_channels, 1, 1), dim=1)
        else:
            self.norm = nn.Identity()
        # assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.pad = ReplicationPad2dDetach((padding,) * 4)
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        basic_module_init(self)

    def forward(self, x):
        x = self.norm(x)
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        x = self.w2(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        return x


class WACBlock(nn.Module):
    """ Window MHA + Multi Layer Conv2d
    """
    def __init__(self, in_channels, num_heads=4, qkv_dim=None, window_size=8, mlp_ratio=2, layer_norm=True):
        super(WACBlock, self).__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        if layer_norm:
            self.norm = RMSNorm1(in_channels)
        else:
            self.norm = None
        self.mha = WindowMHA2d(in_channels, num_heads, qkv_dim=qkv_dim, window_size=window_size)
        self.relative_bias = WindowScoreBias(self.window_size)
        self.conv_mlp = GLUConvMLP(in_channels, in_channels, kernel_size=3, mlp_ratio=mlp_ratio, layer_norm=layer_norm)

    def forward(self, x):
        x1 = self.mha(x, attn_mask=self.relative_bias(), layer_norm=self.norm)
        x = x + x1
        x = x + self.conv_mlp(x)
        return x


class WACBlocks(nn.Module):
    def __init__(self, in_channels, num_heads=4, qkv_dim=None, window_size=8, mlp_ratio=2, num_layers=2, layer_norm=True):
        super(WACBlocks, self).__init__()
        if isinstance(window_size, int):
            window_size = [window_size] * num_layers

        self.blocks = nn.Sequential(
            *[WACBlock(in_channels, window_size=window_size[i],
                       num_heads=num_heads, qkv_dim=qkv_dim, mlp_ratio=mlp_ratio, layer_norm=layer_norm)
              for i in range(num_layers)])

    def forward(self, x):
        return self.blocks(x)


class Overscan(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        C = out_channels * 2 * 2
        self.proj = nn.Conv2d(in_channels * 2 * 2, C, kernel_size=1, stride=1, padding=0)
        self.mha1 = WindowMHA2d(C, num_heads=2, window_size=8)
        self.mha2 = WindowMHA2d(C, num_heads=2, window_size=6)
        self.relative_bias1 = WindowScoreBias(window_size=8)
        self.relative_bias2 = WindowScoreBias(window_size=6)
        self.mlp = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        basic_module_init(self.proj)
        basic_module_init(self.mlp)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        assert x.shape[2] % 8 == 0
        x = self.proj(x)
        x = x + self.mha1(x, attn_mask=self.relative_bias1())
        x = F.pad(x, (-1,) * 4) + self.mlp(x)
        assert x.shape[2] % 6 == 0
        x = x + self.mha2(x, attn_mask=self.relative_bias2())
        x = F.pixel_shuffle(x, 2)
        return x


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        basic_module_init(self.conv)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x), 0.1, inplace=True)
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        icnr_init(self.proj, scale_factor=2)

    def forward(self, x):
        x = F.leaky_relu(self.proj(x), 0.1, inplace=True)
        x = F.pixel_shuffle(x, 2)
        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        # assert in_channels >= out_channels * scale_factor ** 2
        self.proj = nn.Conv2d(in_channels, out_channels * scale_factor ** 2,
                              kernel_size=1, stride=1, padding=0)
        icnr_init(self.proj, scale_factor=scale_factor)

    def forward(self, x):
        x = self.proj(x)
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class WincUNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, base_dim=96,
                 lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 first_layers=2, last_layers=3,
                 scale_factor=2, layer_norm=False, ftf_loss=False):
        super(WincUNetBase, self).__init__()
        assert scale_factor in {1, 2, 4}
        self.ftf_loss = ftf_loss
        C = base_dim
        C2 = int(C * lv2_ratio)
        # assert C % 32 == 0 and C2 % 32 == 0  # slow when C % 32 != 0
        HEADS = 4

        self.overscan = Overscan(in_channels)
        # shallow feature extractor
        self.patch = nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=1, padding=0)
        self.fusion = nn.Conv2d(C // 2 + 16, C, kernel_size=3, stride=1, padding=0)

        # encoder
        self.wac1 = WACBlocks(C, mlp_ratio=lv1_mlp_ratio,
                              window_size=[8, 6] * first_layers, num_heads=HEADS, num_layers=first_layers,
                              layer_norm=layer_norm)
        self.down1 = PatchDown(C, C2)
        self.wac2 = WACBlocks(C2, mlp_ratio=lv2_mlp_ratio,
                              window_size=[8, 6, 8, 6], num_heads=HEADS * 2, num_layers=4,
                              layer_norm=layer_norm)
        # decoder
        self.up1 = PatchUp(C2, C)
        self.wac1_proj = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)
        self.wac3 = WACBlocks(C, mlp_ratio=lv1_mlp_ratio,
                              window_size=[8, 6] * last_layers, num_heads=HEADS, num_layers=last_layers,
                              layer_norm=layer_norm)
        self.to_image = ToImage(C, out_channels, scale_factor=scale_factor)

        basic_module_init(self.patch)
        basic_module_init(self.wac1_proj)
        basic_module_init(self.fusion)

    def forward(self, x):
        ov = self.overscan(x)
        x = self.patch(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = F.pad(x, (-1,) * 4)
        x = torch.cat([x, ov], dim=1)
        x = self.fusion(x)
        x = F.pad(x, (-5,) * 4)
        x = F.leaky_relu(x, 0.1, inplace=True)

        x1 = self.wac1(x)
        x = self.down1(x1)
        x = self.wac2(x)
        x = self.up1(x)
        x = x + self.wac1_proj(x1)
        x = self.wac3(x)
        z = self.to_image(x)

        if self.training and self.ftf_loss:
            # follow the future output in shallow layer
            ftf_loss = F.l1_loss(x1, x.detach())
            return z, ftf_loss * 0.2

        return z


@register_model
class WincUNet1x(I2IBaseModel):
    name = "waifu2x.winc_unet_1x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=64, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 first_layers=2, last_layers=3,
                 **kwargs):
        super(WincUNet1x, self).__init__(locals(), scale=1, offset=8, in_channels=in_channels, blend_size=4)
        self.unet = WincUNetBase(in_channels, out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 first_layers=first_layers, last_layers=last_layers,
                                 scale_factor=1)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


@register_model
class WincUNet2x(I2IBaseModel):
    name = "waifu2x.winc_unet_2x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=96, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 **kwargs):
        super(WincUNet2x, self).__init__(locals(), scale=2, offset=16, in_channels=in_channels, blend_size=8)
        self.unet = WincUNetBase(in_channels, out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 scale_factor=2)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


@register_model
class WincUNet4x(I2IBaseModel):
    name = "waifu2x.winc_unet_4x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=128, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=3,
                 ftf_loss=False,
                 **kwargs):
        super(WincUNet4x, self).__init__(locals(), scale=4, offset=32, in_channels=in_channels, blend_size=16)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = WincUNetBase(in_channels, out_channels=out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 scale_factor=4, ftf_loss=ftf_loss)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)

    def to_2x(self, shared=True):
        unet = self.unet if shared else copy.deepcopy(self.unet)
        return WincUNetDownscaled(unet, downscale_factor=2,
                                  in_channels=self.i2i_in_channels, out_channels=self.out_channels)

    def to_1x(self, shared=True):
        unet = self.unet if shared else copy.deepcopy(self.unet)
        return WincUNetDownscaled(unet=unet, downscale_factor=4,
                                  in_channels=self.i2i_in_channels, out_channels=self.out_channels)


# TODO: Not tested
@register_model
class WincUNetDownscaled(I2IBaseModel):
    name = "waifu2x.winc_unet_downscaled"

    def __init__(self, unet, downscale_factor, in_channels=3, out_channels=3):
        assert downscale_factor in {2, 4}
        offset = 32 // downscale_factor
        scale = 4 // downscale_factor
        blend_size = 4 * downscale_factor
        self.antialias = True
        super().__init__(dict(in_channels=in_channels, out_channels=out_channels,
                              downscale_factor=downscale_factor),
                         scale=scale, offset=offset, in_channels=in_channels, blend_size=blend_size)
        self.unet = unet
        self.downscale_factor = downscale_factor

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            z = F.interpolate(z, size=(z.shape[2] // self.downscale_factor,
                                       z.shape[3] // self.downscale_factor),
                              mode="bicubic", align_corners=False, antialias=self.antialias)
            return z
        else:
            z = torch.clamp(z, 0., 1.)
            z = F.interpolate(z, size=(z.shape[2] // self.downscale_factor,
                                       z.shape[3] // self.downscale_factor),
                              mode="bicubic", align_corners=False, antialias=self.antialias)
            z = torch.clamp(z, 0., 1.)
            return z

    @staticmethod
    def from_4x(unet_4x, downscale_factor):
        net = WincUNetDownscaled(unet=copy.deepcopy(unet_4x.unet),
                                 downscale_factor=downscale_factor,
                                 in_channels=unet_4x.unet.in_channels,
                                 out_channels=unet_4x.unet.out_channels)
        return net


register_model_factory(
    "waifu2x.winc_unet_2xl",
    lambda **kwargs: WincUNet2x(base_dim=96, **kwargs))


register_model_factory(
    "waifu2x.winc_unet_4xl",
    lambda **kwargs: WincUNet4x(base_dim=192, **kwargs))


register_model_factory(
    "waifu2x.winc_unet_1x_small",
    lambda **kwargs: WincUNet1x(base_dim=32, first_layers=1, last_layers=1, lv1_mlp_ratio=1, lv2_mlp_ratio=1, **kwargs))


register_model_factory(
    "waifu2x.winc_unet_4x_ftf",
    lambda **kwargs: WincUNet4x(ftf_loss=True))


def _bench(name, compile):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    model = create_model(name, in_channels=3, out_channels=3).to(device).eval()
    if compile:
        model = torch.compile(model)
    x = torch.zeros((4, 3, 256, 256)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x)
        print(z.shape)
        param = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{param:,}", f"compile={compile}")

    # benchmark
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(100):
            z = model(x)
    print(time.time() - t)


if __name__ == "__main__":
    _bench("waifu2x.winc_unet_1x_small", False)
    _bench("waifu2x.winc_unet_1x", False)
    _bench("waifu2x.winc_unet_2x", False)
    _bench("waifu2x.winc_unet_4x", False)
    _bench("waifu2x.winc_unet_2x", True)
    _bench("waifu2x.winc_unet_4x", True)
