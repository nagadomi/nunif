import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from torchvision.models.swin_transformer import (
    # use SwinTransformer V1
    SwinTransformerBlock as SwinTransformerBlockV1,
)
from torchvision.ops import Permute


def NORM_LAYER(dim):
    # No LayerNorm
    return nn.Identity()


def bchw(bhwc):
    return bhwc.permute(0, 3, 1, 2).contiguous()


def bhwc(bchw):
    return bchw.permute(0, 2, 3, 1).contiguous()


class SwinTransformerBlocks(nn.Module):
    def __init__(self, in_channels, num_head, num_layers, window_size):
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
                    norm_layer=NORM_LAYER,
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
        x = bchw(x)
        x = self.conv(x)
        x = bhwc(x)
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
        x = bchw(x)
        x = F.pixel_shuffle(x, 2)
        x = bhwc(x)
        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.to_4x = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_channels, out_channels * 2 ** 4),
            Permute([0, 3, 1, 2]),
            nn.PixelShuffle(4))
        self.to_2x = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_channels, out_channels * 2 ** 2),
            Permute([0, 3, 1, 2]),
            nn.PixelShuffle(2))
        self.to_1x = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_channels, out_channels),
            Permute([0, 3, 1, 2]))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, scale=None):
        if scale is None or scale == 1:
            x1 = self.to_1x(x)
        else:
            x1 = None
        if scale is None or scale == 2:
            x2 = self.to_2x(x)
        else:
            x2 = None
        if scale is None or scale == 4:
            x4 = self.to_4x(x)
        else:
            x4 = None

        return x1, x2, x4


class SwinUNetBase(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_dim=96, base_layers=2):
        super().__init__()
        assert base_dim % 16 == 0 and base_dim % 6 == 0
        assert base_layers % 2 == 0
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
        self.swin1 = SwinTransformerBlocks(C, num_head=H, num_layers=L, window_size=W)
        self.down1 = PatchDown(C, C * 2)
        self.swin2 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
        self.down2 = PatchDown(C * 2, C * 2)
        self.swin3 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L * 3, window_size=W)
        self.up2 = PatchUp(C * 2, C * 2)
        self.proj1 = nn.Linear(C, C * 2)
        self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
        self.up1 = PatchUp(C * 2, C * 2)
        self.swin5 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
        self.to_image = ToImage(C * 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for m in (list(self.patch.modules()) + [self.proj1]):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, scale=None):
        x2 = self.patch(x)
        x2 = F.pad(x2, (-6, -6, -6, -6))
        assert x2.shape[2] % 12 == 0 and x2.shape[2] % 16 == 0
        x2 = bhwc(x2)

        x3 = self.swin1(x2)
        x4 = self.down1(x3)
        x4 = self.swin2(x4)
        x5 = self.down2(x4)
        x5 = self.swin3(x5)
        x5 = self.up2(x5)
        x = x5 + x4
        x = self.swin4(x)
        x = self.up1(x)
        x = x + self.proj1(x3)
        x = self.swin5(x)
        x1, x2, x4 = self.to_image(x, scale=scale)

        return x1, x2, x4


@register_model
class SwinUNetUnif(I2IBaseModel):
    name = "waifu2x.swin_unet_unif"

    def __init__(self, in_channels=3, out_channels=3, scale=4, offset=32, blend_size=4, unet=None):
        super().__init__(locals(), in_channels=in_channels,
                         scale=scale, offset=offset,
                         blend_size=blend_size)
        self.out_channels = out_channels
        if unet is not None:
            self.unet = unet
        else:
            self.unet = SwinUNetBase(
                in_channels=in_channels,
                out_channels=out_channels,
                base_dim=96, base_layers=2)

    def get_config(self):
        config = dict(super().get_config())
        config.update({
            "i2i_unif_scale_factors": [1, 2, 4],
            "i2i_unif_model_offsets": [8, 16, 32],
        })
        return config

    def _forward(self, x, scale=None):
        x1, x2, x4 = self.unet(x, scale=scale)
        return x1, x2, x4

    def forward(self, x, scale_factor):
        x1, x2, x4 = self._forward(x, scale_factor)
        if scale_factor == 1:
            z = x1
        elif scale_factor == 2:
            z = x2
        elif scale_factor == 4:
            z = x4
        else:
            raise ValueError("scale_factor")

        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)

    def to_4x(self):
        net = SwinUNetUnif4x(
            in_channels=self.i2i_in_channels, out_channels=self.out_channels,
            unet=self.unet
        )
        return net

    def to_2x(self):
        net = SwinUNetUnif2x(
            in_channels=self.i2i_in_channels, out_channels=self.out_channels,
            unet=self.unet
        )
        return net

    def to_1x(self):
        net = SwinUNetUnif1x(
            in_channels=self.i2i_in_channels, out_channels=self.out_channels,
            unet=self.unet
        )
        return net

    def copy_from_swin_unet_4x(self, model):
        self.unet.patch.load_state_dict(model.unet.patch.state_dict())
        self.unet.swin1.load_state_dict(model.unet.swin1.state_dict())
        self.unet.swin2.load_state_dict(model.unet.swin2.state_dict())
        self.unet.swin3.load_state_dict(model.unet.swin3.state_dict())
        self.unet.swin4.load_state_dict(model.unet.swin4.state_dict())
        self.unet.swin5.load_state_dict(model.unet.swin5.state_dict())
        self.unet.down1.load_state_dict(model.unet.down1.state_dict())
        self.unet.down2.load_state_dict(model.unet.down2.state_dict())
        self.unet.up1.load_state_dict(model.unet.up1.state_dict())
        self.unet.up2.load_state_dict(model.unet.up2.state_dict())
        self.unet.proj1.load_state_dict(model.unet.proj2.state_dict())

    def freeze(self):
        for m in self.parameters():
            m.requires_grad = False
        for m in self.unet.to_image.parameters():
            m.requires_grad = True


class SwinUNetUnif1x(SwinUNetUnif):
    name = "waifu2x.swin_unet_unif_1x"

    def __init__(self, in_channels=3, out_channels=3, unet=None):
        super().__init__(scale=1, offset=8, in_channels=in_channels, blend_size=4, unet=unet)

    def forward(self, x):
        x1, x2, x4 = self._forward(x, scale=1)
        return torch.clamp(x1, 0, 1)


class SwinUNetUnif2x(SwinUNetUnif):
    name = "waifu2x.swin_unet_unif_2x"

    def __init__(self, in_channels=3, out_channels=3, unet=None):
        super().__init__(scale=2, offset=16, in_channels=in_channels, blend_size=4, unet=unet)

    def forward(self, x):
        x1, x2, x4 = self._forward(x, scale=2)
        return torch.clamp(x2, 0, 1)


class SwinUNetUnif4x(SwinUNetUnif):
    name = "waifu2x.swin_unet_unif_4x"

    def __init__(self, in_channels=3, out_channels=3, unet=None):
        super().__init__(scale=4, offset=32, in_channels=in_channels, blend_size=4, unet=unet)

    def forward(self, x):
        x1, x2, x4 = self._forward(x, scale=4)
        return torch.clamp(x4, 0, 1)


def _test():
    import io
    device = "cuda:0"
    model = SwinUNetUnif(in_channels=3, out_channels=3).to(device)
    net1 = model.to_1x()
    net2 = model.to_2x()
    net4 = model.to_4x()

    x = torch.zeros((1, 3, 64, 64)).to(device)
    with torch.no_grad():
        x1 = model(x, 1)
        x2 = model(x, 2)
        x4 = model(x, 4)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        print(model.name, "output", x1.shape, x2.shape, x4.shape, "size", len(buf.getvalue()))
        print("net", net1(x).shape, net2(x).shape, net4(x).shape)


def _copy_test():
    from nunif.models import load_model, save_model

    model_old, _ = load_model("waifu2x/pretrained_models/swin_unet/art/scale4x.pth")
    model = SwinUNetUnif()
    model.copy_from_swin_unet_4x(model_old)
    save_model(model, "./models/unif_copy/unif4x.pth")
    for noise_level in range(4):
        model_old, _ = load_model(f"waifu2x/pretrained_models/swin_unet/art/noise{noise_level}_scale4x.pth")
        model = SwinUNetUnif()
        model.copy_from_swin_unet_4x(model_old)
        save_model(model, f"./models/unif_copy/noise{noise_level}_unif4x.pth")


def _convert_tool_main():
    import argparse
    from nunif.models import load_model, save_model

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input swin_unet_4x model")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output swin_unet_unif model")
    args = parser.parse_args()

    model_old, _ = load_model(args.input)
    model = SwinUNetUnif()
    model.copy_from_swin_unet_4x(model_old)
    save_model(model, args.output)


if __name__ == "__main__":
    _convert_tool_main()
