import torch
import torch.nn as nn
from nunif.models import I2IBaseModel, register_model


@register_model
class UpConv7(I2IBaseModel):
    name = "waifu2x.upconv_7"

    def __init__(self, in_channels=3, out_channels=3):
        super(UpConv7, self).__init__(locals(), scale=2, offset=14, in_channels=in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, out_channels, 4, 2, 3),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.net(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


if __name__ == "__main__":
    import torch
    from nunif.models import create_model
    device = "cuda:0"
    model = create_model(UpConv7.name, in_channels=3, out_channels=3).to(device)
    print(model)
    x = torch.zeros((1, 3, 256, 256)).to(device)
    with torch.no_grad():
        z = model(x)
        print(z.shape)
        print(model.name, model.i2i_offset, model.i2i_scale)
