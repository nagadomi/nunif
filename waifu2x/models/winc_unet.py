import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model, register_model_factory
from nunif.modules.attention import WindowMHA2d


class WindowBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size = [window_size, window_size]
        self.window_size = window_size

        index, unique_delta = self._gen_input(self.window_size)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)
        if hidden_dim is None:
            hidden_dim = int((self.window_size[0] * self.window_size[1]) ** 0.5) * 2

        self.to_bias = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _gen_input(window_size):
        N = window_size[0] * window_size[1]
        mesh_y, mesh_x = torch.meshgrid(torch.arange(0, window_size[0]),
                                        torch.arange(0, window_size[1]), indexing="ij")
        positions = torch.stack((mesh_y, mesh_x), dim=2).reshape(N, 2)
        delta = torch.cat([positions[i].view(1, 2) - positions
                           for i in range(positions.shape[0])], dim=0)
        delta = [tuple(p) for p in delta.tolist()]
        unique_delta = sorted(list(set(delta)))
        index = [unique_delta.index(d) for d in delta]
        index = torch.tensor(index, dtype=torch.int64)
        unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
        unique_delta = unique_delta / unique_delta.abs().max()
        return index, unique_delta

    def forward(self):
        N = self.window_size[0] * self.window_size[1]
        bias = self.to_bias(self.delta)
        # (N,N) float attention score bias
        bias = bias[self.index].reshape(N, N)
        return bias


class GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.w1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=kernel_size, stride=1,
                            padding=padding, padding_mode="replicate")

    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.w2(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        return x


class WACBlock(nn.Module):
    """ Window MHA + Multi Layer Conv2d
    """
    def __init__(self, in_channels, num_heads=4, qkv_dim=16, window_size=8):
        super(WACBlock, self).__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.norm = nn.LayerNorm(in_channels, bias=False)
        self.mha = WindowMHA2d(in_channels, num_heads, qkv_dim=qkv_dim, window_size=window_size)
        self.relative_bias = WindowBias(self.window_size)
        self.conv_mlp = GLUConvMLP(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.relative_bias(), layer_norm=self.norm)
        x = x + self.conv_mlp(x)
        return x


class WACBlocks(nn.Module):
    def __init__(self, in_channels, num_heads=4, qkv_dim=16, window_size=8, num_layers=2):
        super(WACBlocks, self).__init__()
        if isinstance(window_size, int):
            window_size = [window_size] * num_layers

        self.blocks = nn.Sequential(
            *[WACBlock(in_channels, window_size=window_size[i], num_heads=num_heads, qkv_dim=qkv_dim)
              for i in range(num_layers)])

    def forward(self, x):
        return self.blocks(x)


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x), 0.1, inplace=True)
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0)

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

    def forward(self, x):
        x = self.proj(x)
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class WincUNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, base_dim=96, add_dim=0, scale_factor=2,
                 lv3_window_size=[6, 12, 12, 6], lv3_dim_scale=2):
        super(WincUNetBase, self).__init__()
        assert scale_factor in {1, 2, 4}
        C = base_dim
        HEADS = 4
        QKV_DIM = C // 4

        # shallow feature extractor
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # encoder
        self.wac1 = WACBlocks(C, num_heads=HEADS, qkv_dim=QKV_DIM)
        self.down1 = PatchDown(C, C * 2)
        self.wac2 = WACBlocks(C * 2, num_heads=HEADS * 2, qkv_dim=QKV_DIM)
        self.down2 = PatchDown(C * 2, C * lv3_dim_scale + add_dim)
        self.wac3 = WACBlocks(C * lv3_dim_scale + add_dim, window_size=lv3_window_size, num_heads=HEADS * 2,
                              qkv_dim=QKV_DIM, num_layers=4)
        # decoder
        self.up2 = PatchUp(C * lv3_dim_scale + add_dim, C * 2 + add_dim)
        self.wac2_proj = nn.Conv2d(C * 2, C * 2 + add_dim, kernel_size=1, stride=1, padding=0)
        self.wac4 = WACBlocks(C * 2 + add_dim, num_heads=HEADS * 2, qkv_dim=QKV_DIM, num_layers=3)
        self.up1 = PatchUp(C * 2 + add_dim, C + add_dim)
        self.wac1_proj = nn.Conv2d(C, C + add_dim, kernel_size=1, stride=1, padding=0)
        self.wac5 = WACBlocks(C + add_dim, num_heads=HEADS, qkv_dim=QKV_DIM, num_layers=3)
        self.to_image = ToImage(C + add_dim, out_channels, scale_factor=scale_factor)

    def forward(self, x):
        x = self.patch(x)
        x = F.pad(x, (-6, -6, -6, -6))

        x1 = self.wac1(x)
        x2 = self.down1(x1)
        x2 = self.wac2(x2)
        x3 = self.down2(x2)
        x3 = self.wac3(x3)
        x3 = self.up2(x3)
        x = x3 + self.wac2_proj(x2)
        x = self.wac4(x)
        x = self.up1(x)
        x = x + self.wac1_proj(x1)
        x = self.wac5(x)
        z = self.to_image(x)
        return z


@register_model
class WincUNet2x(I2IBaseModel):
    name = "waifu2x.winc_unet_2x"

    def __init__(self, in_channels=3, out_channels=3, base_dim=96, lv3_window_size=[6, 12, 12, 6], lv3_dim_scale=2):
        super(WincUNet2x, self).__init__(locals(), scale=2, offset=16, in_channels=in_channels, blend_size=8)
        self.unet = WincUNetBase(
            in_channels, out_channels,
            base_dim=base_dim, add_dim=base_dim // 4,
            lv3_window_size=lv3_window_size,
            lv3_dim_scale=lv3_dim_scale,
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

    def __init__(self, in_channels=3, out_channels=3, base_dim=96, lv3_window_size=[6, 6, 6, 6], lv3_dim_scale=2):
        super(WincUNet4x, self).__init__(locals(), scale=4, offset=32, in_channels=in_channels, blend_size=16)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = WincUNetBase(
            in_channels, out_channels=out_channels,
            base_dim=base_dim, add_dim=base_dim,
            lv3_window_size=lv3_window_size,
            lv3_dim_scale=lv3_dim_scale,
            scale_factor=4)

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
    lambda **kwargs: WincUNet2x(base_dim=192, lv3_dim_scale=4, **kwargs))

register_model_factory(
    "waifu2x.winc_unet_4xl",
    lambda **kwargs: WincUNet4x(base_dim=192, lv3_dim_scale=4, **kwargs))


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
        print(model.name, model.i2i_offset, model.i2i_scale, f"compile={compile}")

    # benchmark
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(100):
            z = model(x)
    print(time.time() - t)


if __name__ == "__main__":
    _bench("waifu2x.winc_unet_2x", False)
    _bench("waifu2x.winc_unet_4x", False)
    _bench("waifu2x.winc_unet_2x", True)
    _bench("waifu2x.winc_unet_4x", True)
