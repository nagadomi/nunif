import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from nunif.models import I2IBaseModel, register_model, register_model_builder
from nunif.modules.norm import LayerNormNoBias
from torchvision.models.swin_transformer import (
    # use SwinTransformer V1
    SwinTransformerBlock as SwinTransformerBlockV1,
)


# No LayerNorm
def NO_NORM_LAYER(dim):
    return nn.Identity()


class SwinTransformerBlocks(nn.Module):
    def __init__(self, in_channels, num_head, num_layers, window_size, norm_layer=NO_NORM_LAYER):
        super().__init__()
        layers = []
        for i_layer in range(num_layers):
            layers.append(
                SwinTransformerBlockV1(
                    in_channels,
                    num_head,
                    window_size=window_size,
                    shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=2.,
                    dropout=0.,
                    attention_dropout=0.,
                    stochastic_depth_prob=0.,
                    norm_layer=norm_layer,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        z = self.block(x)
        return z


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC->BCHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW->BHWC
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels * 4)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)  # BHWC->BCHW
        x = F.pixel_shuffle(x, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW->BHWC
        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        assert scale_factor in {1, 2, 4, 8}
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        if scale_factor == 1:
            self.proj = nn.Linear(in_channels, out_channels)
        elif scale_factor in {2, 4}:
            scale2 = scale_factor ** 2
            self.proj = nn.Linear(in_channels, out_channels * scale2)
        elif scale_factor in {8}:
            scale2 = scale_factor ** 2
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_channels * scale2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(out_channels * scale2, out_channels * scale2))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class SwinUNetBase(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=1,
                 norm_layer=NO_NORM_LAYER):
        super().__init__()
        assert scale_factor in {1, 2, 4, 8}
        assert base_dim % 16 == 0 and base_dim % 6 == 0
        assert base_layers % 2 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        C = base_dim
        H = C // 16
        L = base_layers
        W = [6, 6]
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.swin1 = SwinTransformerBlocks(C, num_head=H, num_layers=L, window_size=W,
                                           norm_layer=norm_layer)
        self.down1 = PatchDown(C, C * 2)
        self.swin2 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W,
                                           norm_layer=norm_layer)
        self.down2 = PatchDown(C * 2, C * 2)
        self.swin3 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L * 3, window_size=W,
                                           norm_layer=norm_layer)
        if scale_factor in {1, 2}:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Identity()
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W,
                                               norm_layer=norm_layer)
            self.up1 = PatchUp(C * 2, C)
            self.swin5 = SwinTransformerBlocks(C, num_head=H, num_layers=L, window_size=W,
                                               norm_layer=norm_layer)
            self.to_image = ToImage(C, out_channels, scale_factor=scale_factor)
        elif scale_factor in {4, 8}:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Linear(C, C * 2)
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W,
                                               norm_layer=norm_layer)
            self.up1 = PatchUp(C * 2, C * 2)
            self.swin5 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W,
                                               norm_layer=norm_layer)
            self.to_image = ToImage(C * 2, out_channels, scale_factor=scale_factor)

        self.reset_parameters()

    def reset_parameters(self):
        for m in (list(self.patch.modules()) + [self.proj1, self.proj2]):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x2 = self.patch(x)
        x2 = F.pad(x2, (-6, -6, -6, -6))
        assert x2.shape[2] % 12 == 0 and x2.shape[2] % 16 == 0
        x2 = x2.permute(0, 2, 3, 1).contiguous()  # BHWC

        x3 = self.swin1(x2)
        x4 = self.down1(x3)
        x4 = self.swin2(x4)
        x5 = self.down2(x4)
        x5 = self.swin3(x5)
        x5 = self.up2(x5)
        x = x5 + self.proj1(x4)
        x = self.swin4(x)
        x = self.up1(x)
        x = x + self.proj2(x3)
        x = self.swin5(x)
        x = self.to_image(x)

        return x


@register_model
class SwinUNet(I2IBaseModel):
    name = "waifu2x.swin_unet_1x"

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=1, offset=8, in_channels=in_channels, blend_size=4)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96, base_layers=2,
            scale_factor=1)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


@register_model
class SwinUNet2x(I2IBaseModel):
    name = "waifu2x.swin_unet_2x"

    def __init__(self, in_channels=3, out_channels=3, base_dim=96, layer_norm=False):
        super().__init__(locals(), scale=2, offset=16, in_channels=in_channels, blend_size=8)
        norm_layer = LayerNormNoBias if layer_norm else NO_NORM_LAYER
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=base_dim, base_layers=2,
            norm_layer=norm_layer,
            scale_factor=2)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


def resize_antialias(x, antialias):
    B, C, H, W = x.shape
    x = F.interpolate(x, size=(H * 2, W * 2), mode="bicubic",
                      align_corners=False, antialias=antialias)
    x = F.interpolate(x, size=(H, W), mode="bicubic",
                      align_corners=False, antialias=antialias)
    return x


@register_model
class SwinUNet4x(I2IBaseModel):
    name = "waifu2x.swin_unet_4x"

    def __init__(self, in_channels=3, out_channels=3, pre_antialias=False,
                 base_dim=96, layer_norm=False):
        super().__init__(locals(), scale=4, offset=32, in_channels=in_channels, blend_size=16)
        self.out_channels = out_channels
        self.pre_antialias = pre_antialias
        self.antialias = True
        norm_layer = LayerNormNoBias if layer_norm else NO_NORM_LAYER
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=base_dim, base_layers=2,
            norm_layer=norm_layer,
            scale_factor=4)

    def forward(self, x):
        if self.pre_antialias:
            x = resize_antialias(x, antialias=self.antialias)
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)

    def to_2x(self, shared=True):
        if shared:
            unet = self.unet
        else:
            unet = copy.deepcopy(self.unet)
        return SwinUNetDownscaled(in_channels=self.i2i_in_channels, out_channels=self.out_channels,
                                  downscale_factor=2, unet=unet, pre_antialias=self.pre_antialias)

    def to_1x(self, shared=True):
        if shared:
            unet = self.unet
        else:
            unet = copy.deepcopy(self.unet)
        return SwinUNetDownscaled(in_channels=self.i2i_in_channels, out_channels=self.out_channels,
                                  downscale_factor=4, unet=unet, pre_antialias=self.pre_antialias)


@register_model
class SwinUNet8x(I2IBaseModel):
    name = "waifu2x.swin_unet_8x"

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=4, offset=64, in_channels=in_channels, blend_size=32)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96, base_layers=2,
            scale_factor=8)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


@register_model
class SwinUNetDownscaled(I2IBaseModel):
    name = "waifu2x.swin_unet_downscaled"

    def __init__(self, in_channels=3, out_channels=3, downscale_factor=2, unet=None, pre_antialias=False):
        assert downscale_factor in {2, 4}
        offset = 32 // downscale_factor
        scale = 4 // downscale_factor
        blend_size = 4 * downscale_factor
        super().__init__(dict(in_channels=in_channels, out_channels=out_channels,
                              downscale_factor=downscale_factor),
                         scale=scale, offset=offset, in_channels=in_channels, blend_size=blend_size)
        if unet is None:
            self.unet = SwinUNetBase(
                in_channels=in_channels,
                out_channels=out_channels,
                base_dim=96, base_layers=2,
                scale_factor=4)
        else:
            self.unet = unet
        self.antialias = True
        self.pre_antialias = pre_antialias
        self.downscale_factor = downscale_factor

    def forward(self, x):
        if self.pre_antialias:
            x = resize_antialias(x, antialias=self.antialias)
        z = self.unet(x)
        if self.training:
            z = F.interpolate(z, size=(z.shape[2] // self.downscale_factor, z.shape[3] // self.downscale_factor),
                              mode="bicubic", align_corners=False, antialias=self.antialias)
            return z
        else:
            z = torch.clamp(z, 0., 1.)
            z = F.interpolate(z, size=(z.shape[2] // self.downscale_factor, z.shape[3] // self.downscale_factor),
                              mode="bicubic", align_corners=False, antialias=self.antialias)
            z = torch.clamp(z, 0., 1.)
            return z

    @staticmethod
    def from_4x(swin_unet_4x, downscale_factor):
        net = SwinUNetDownscaled(in_channels=swin_unet_4x.unet.in_channels,
                                 out_channels=swin_unet_4x.unet.out_channels,
                                 downscale_factor=downscale_factor,
                                 unet=copy.deepcopy(swin_unet_4x.unet))
        return net


def swin_unet_4xl(**kwargs):
    return SwinUNet4x(base_dim=192, layer_norm=True, **kwargs)


register_model_builder("waifu2x.swin_unet_4xl", swin_unet_4xl)


def _test():
    import io
    device = "cuda:0"
    for model in (SwinUNet(in_channels=3, out_channels=3),
                  SwinUNet2x(in_channels=3, out_channels=3),
                  SwinUNet4x(in_channels=3, out_channels=3),
                  SwinUNetDownscaled(in_channels=3, out_channels=3, downscale_factor=2),
                  SwinUNetDownscaled(in_channels=3, out_channels=3, downscale_factor=4)):
        model = model.to(device)
        # Note: input size must be `(SIZE - 16) % 12 == 0 and (SIZE - 16) % 16 == 0`,
        # e.g. 64,112,160,256,400,640,1024
        x = torch.zeros((1, 3, 64, 64)).to(device)
        with torch.no_grad():
            z = model(x)
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            print(model.name, "output", z.shape, "size", len(buf.getvalue()))


def _convert_tool_main():
    import argparse
    from nunif.models import load_model, save_model

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input 4x model")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output swin_unet_downscaled_model")
    parser.add_argument("--scale", type=int, choices=[1, 2], required=True,
                        help="scale factor for output swin_unet_model")
    args = parser.parse_args()
    model_4x, _ = load_model(args.input)
    if args.scale == 2:
        model = model_4x.to_2x()
    elif args.scale == 1:
        model = model_4x.to_1x()

    save_model(model, args.output)


if __name__ == "__main__":
    if False:
        _test()
    else:
        _convert_tool_main()
