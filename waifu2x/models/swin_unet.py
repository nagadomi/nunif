import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from torchvision.models.swin_transformer import (
    # use SwinTransformer V1
    SwinTransformerBlock as SwinTransformerBlockV1,
)


# No LayerNorm
def NORM_LAYER(dim):
    return nn.Identity()


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
        assert scale_factor in {1, 2, 4}
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        if scale_factor == 1:
            self.proj = nn.Linear(in_channels, out_channels)
        elif scale_factor in {2, 4}:
            scale2 = scale_factor ** 2
            self.proj = nn.Linear(in_channels, out_channels * scale2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class SwinUNetBase(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=1):
        super().__init__()
        assert scale_factor in {1, 2, 4}
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
        if scale_factor in {1, 2}:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Identity()
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
            self.up1 = PatchUp(C * 2, C)
            self.swin5 = SwinTransformerBlocks(C, num_head=H, num_layers=L, window_size=W)
            self.to_image = ToImage(C, out_channels, scale_factor=scale_factor)
        elif scale_factor == 4:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Linear(C, C * 2)
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
            self.up1 = PatchUp(C * 2, C * 2)
            self.swin5 = SwinTransformerBlocks(C * 2, num_head=H, num_layers=L, window_size=W)
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
class UpSwinUNet(I2IBaseModel):
    name = "waifu2x.swin_unet_2x"

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=2, offset=16, in_channels=in_channels, blend_size=4)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96, base_layers=2,
            scale_factor=2)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


@register_model
class UpSwinUNet4x(I2IBaseModel):
    name = "waifu2x.swin_unet_4x"

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(locals(), scale=4, offset=32, in_channels=in_channels, blend_size=4)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96, base_layers=2,
            scale_factor=4)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0, 1)


if __name__ == "__main__":
    import io
    device = "cuda:0"
    for model in (SwinUNet(in_channels=3, out_channels=3),
                  UpSwinUNet(in_channels=3, out_channels=3),
                  UpSwinUNet4x(in_channels=3, out_channels=3)):
        model = model.to(device)
        # Note: input size must be `(SIZE - 16) % 12 == 0 and (SIZE - 16) % 16 == 0`,
        # e.g. 64,112,160,256,400,640,1024
        x = torch.zeros((1, 3, 64, 64)).to(device)
        with torch.no_grad():
            z = model(x)
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            print(model.name, "output", z.shape, "size", len(buf.getvalue()))
