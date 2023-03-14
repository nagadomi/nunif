import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Permute
from nunif.models import I2IBaseModel, register_model
from .swin_unet import SwinUNetBase


def bchw(bhwc):
    return bhwc.permute(0, 3, 1, 2).contiguous()


def bhwc(bchw):
    return bchw.permute(0, 2, 3, 1).contiguous()


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.to_4x = nn.Sequential(
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

    def forward(self, x, scale_factor=4):
        if scale_factor == 1:
            return self.to_1x(x)
        elif scale_factor == 2:
            return self.to_2x(x)
        else:
            return self.to_4x(x)


@register_model
class SwinUNetUnif(I2IBaseModel):
    name = "waifu2x.swin_unet_unif"

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(dict(in_channels=in_channels, out_channels=out_channels),
                         in_channels=in_channels, scale=4, offset=32, blend_size=4)
        self.unet = SwinUNetBase(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=96, base_layers=2,
            scale_factor=4)
        self.unet.to_image = nn.Identity()
        self.to_image = ToImage(96 * 2, out_channels)
        self.scale_factor = 4
        self.out_channels = out_channels

    def forward(self, x, scale_factor=None):
        scale_factor = scale_factor or self.scale_factor
        z = self.to_image(self.unet(x), scale_factor=scale_factor)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)

    def to_4x(self):
        return SwinUNetUnif4x(unet=self, 
                              in_channels=self.i2i_in_channels, 
                              out_channels=self.out_channels)

    def to_2x(self):
        return SwinUNetUnif2x(unet=self,
                              in_channels=self.i2i_in_channels,
                              out_channels=self.out_channels)

    def to_1x(self):
        return SwinUNetUnif1x(unet=self,
                              in_channels=self.i2i_in_channels, 
                              out_channels=self.out_channels)

    def freeze(self):
        for m in self.parameters():
            m.requires_grad = False
        for m in self.to_image.parameters():
            m.requires_grad = True

    @staticmethod
    def from_4x(model_4x):
        net = SwinUNetUnif()
        net.unet = copy.deepcopy(model_4x.unet)
        org_to_image = net.unet.to_image
        net.unet.to_image = nn.Identity()
        net.to_image = ToImage(96 * 2, 3)
        net.to_image.to_4x[0].load_state_dict(org_to_image.proj.state_dict())
        return net


class SwinUNetUnif1x(I2IBaseModel):
    name = "waifu2x.swin_unet_unif_1x"

    def __init__(self, unet, in_channels=3, out_channels=3):
        super().__init__({}, scale=1, offset=8, in_channels=in_channels, blend_size=4)
        self.unet = unet

    def forward(self, x):
        return self.unet(x, scale_factor=1)


class SwinUNetUnif2x(I2IBaseModel):
    name = "waifu2x.swin_unet_unif_2x"

    def __init__(self, unet, in_channels=3, out_channels=3):
        super().__init__({}, scale=2, offset=16, in_channels=in_channels, blend_size=4)
        self.unet = unet

    def forward(self, x):
        return self.unet(x, scale_factor=2)


class SwinUNetUnif4x(I2IBaseModel):
    name = "waifu2x.swin_unet_unif_4x"

    def __init__(self, unet, in_channels=3, out_channels=3):
        super().__init__({}, scale=4, offset=32, in_channels=in_channels, blend_size=4)
        self.unet = unet

    def forward(self, x):
        return self.unet(x, scale_factor=4)


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

    model_4x, _ = load_model(args.input)
    model = SwinUNetUnif.from_4x(model_4x)
    save_model(model, args.output)


if __name__ == "__main__":
    # _test()
    _convert_tool_main()
