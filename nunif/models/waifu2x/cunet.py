import torch.nn as nn
import torch.nn.functional as F
from .. model import Model
from ... modules import InplaceClip, SEBlock
from ... modules import functional as NF


INPLACE=True



class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=INPLACE),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=INPLACE),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=INPLACE)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=INPLACE)

        x1 = F.pad(x1, (-4, -4, -4, -4), mode='constant')
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=INPLACE)
        z = self.conv_bottom(x3)
        return z


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=INPLACE)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=INPLACE)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=INPLACE)

        x2 = F.pad(x2, (-4, -4, -4, -4), mode='constant')
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=INPLACE)

        x1 = F.pad(x1, (-16, -16, -16, -16), mode='constant')
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=INPLACE)

        z = self.conv_bottom(x5)
        return z


class UpCUNet(Model):
    name = "waifu2x.upcunet"
    def __init__(self, in_channels, out_channels):
        super(UpCUNet, self).__init__(UpCUNet.name, in_channels=in_channels,
                                      out_channels=out_channels, scale=2, offset=36)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)
        self.register_kwargs({"in_channels": in_channels, "out_channels": out_channels})

    def forward(self, x):
        z1 = self.unet1(x)
        z1 = NF.inplace_clip(z1, 0.0, 1.0)
        z2 = self.unet2(z1)
        z1 = F.pad(z1, (-20, -20, -20, -20), mode='constant')
        z = z1 + z2
        z = NF.inplace_clip(z, 0.0, 1.0)
        if self.training:
            return (z, z1)
        else:
            return z1


class CUNet(Model):
    name = "waifu2x.cunet"

    def __init__(self, in_channels, out_channels):
        super(CUNet, self).__init__(CUNet.name, in_channels=in_channels,
                                    out_channels=out_channels, scale=1, offset=28)
        self.unet1 = UNet1(in_channels, out_channels, deconv=False)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)
        self.register_kwargs({"in_channels": in_channels, "out_channels": out_channels})

    def forward(self, x):
        z1 = self.unet1(x)
        z1 = NF.inplace_clip(z1, 0.0, 1.0)
        z2 = self.unet2(z1)
        z1 = F.pad(z1, (-20, -20, -20, -20), mode='constant')
        z = z1 + z2
        z = NF.inplace_clip(z, 0.0, 1.0)

        if self.training:
            return (z, z1)
        else:
            return z1


if __name__ == "__main__":
    import torch
    device = "cuda:0"
    model = CUNet(in_channels=3, out_channels=3).to(device)
    print(model)
    x = torch.zeros((1, 3, 256, 256)).to(device)
    with torch.no_grad():
        z, z1 = model(x)
        print(z.shape, z1.shape)
