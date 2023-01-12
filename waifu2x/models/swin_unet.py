import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from torchvision.models.swin_transformer import (
    # use SwinTransformer V1
    SwinTransformerBlock as SwinTransformerBlockV1,
    PatchMerging as PatchMergingV1
)


# No LayerNorm
NORM_LAYER = lambda dim: nn.Identity()


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
                    mlp_ratio=2.0,
                    dropout=0,
                    attention_dropout=0,
                    stochastic_depth_prob=0,
                    norm_layer=NORM_LAYER,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def PatchDown(in_channels):
    return PatchMergingV1(in_channels, norm_layer=NORM_LAYER)


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels * 4)

    def forward(self, x):
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)  # BHWC->BCHW
        x = F.pixel_shuffle(x, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW->BHWC
        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        assert scale_factor in {1, 2}
        self.scale_factor = scale_factor
        if scale_factor == 1:
            self.linear = nn.Linear(in_channels, out_channels)
        elif scale_factor == 2:
            assert in_channels % 4 == 0
            self.linear = nn.Linear(in_channels, out_channels * 4)

    def forward(self, x):
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.scale_factor == 2:
            x = F.pixel_shuffle(x, 2)
        return x


class SwinUNetBase(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_dim=96, scale_factor=1):
        super().__init__()
        assert base_dim % 16 == 0 and base_dim % 6 == 0
        C = base_dim
        H = C // 16
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.swin1 = SwinTransformerBlocks(C, num_head=H, num_layers=2, window_size=[6, 6])
        self.swin2 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=2, window_size=[6, 6])
        self.swin3 = SwinTransformerBlocks(C * 4, num_head=H, num_layers=6, window_size=[6, 6])
        self.down1 = PatchDown(C)
        self.down2 = PatchDown(C * 2)
        self.up2 = PatchUp(C * 4, C * 2)
        self.up1 = PatchUp(C * 2, C)
        self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=2, window_size=[6, 6])
        self.swin5 = SwinTransformerBlocks(C, num_head=H, num_layers=2, window_size=[6, 6])
        self.to_image = ToImage(C, out_channels, scale_factor=scale_factor)

    def forward(self, x):
        x2 = self.patch(x)
        x2 = F.pad(x2, (-6, -6, -6, -6))

        assert x2.shape[2] % 12 == 0 and x2.shape[2] % 16 == 0

        x2 = x2.permute(0, 2, 3, 1).contiguous()  # BHWC
        x2 = self.swin1(x2)
        x3 = self.down1(x2)
        x3 = self.swin2(x3)
        x4 = self.down2(x3)
        x4 = self.swin3(x4)
        x4 = self.up2(x4)

        x = x4 + x3
        x = self.swin4(x)
        x = self.up1(x)
        x = x + x2
        x = self.swin5(x)
        x = self.to_image(x)

        return x


@register_model
class SwinUNet(I2IBaseModel):
    name = "waifu2x.swin_unet_1x"
    name_alias = ("waifu2x.swinunet",)

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=1, offset=8, in_channels=in_channels)
        self.unet = SwinUNetBase(
            in_channels=in_channels, 
            out_channels=out_channels,
            base_dim=96,
            scale_factor=1)

    def forward(self, x):
        assert x.shape[2] % 64 == 0
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


@register_model
class UpSwinUNet(I2IBaseModel):
    name = "waifu2x.swin_unet_2x"
    name_alias = ("waifu2x.upswinunet",)

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=2, offset=16, in_channels=in_channels)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96,
            scale_factor=2)

    def forward(self, x):
        assert x.shape[2] % 64 == 0
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


if __name__ == "__main__":
    device = "cuda:0"
    model = UpSwinUNet(in_channels=3, out_channels=3).to(device)
    x = torch.zeros((1, 3, 64, 64)).to(device)
    with torch.no_grad():
        z = model(x)
        print(z.shape)
