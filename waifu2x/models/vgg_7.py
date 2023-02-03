import torch
import torch.nn as nn
from nunif.models import I2IBaseModel, register_model


@register_model
class VGG7(I2IBaseModel):
    name = "waifu2x.vgg_7"

    def __init__(self, in_channels=3, out_channels=3):
        super(VGG7, self).__init__(locals(), scale=1, offset=7, in_channels=in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, out_channels, 3, 1, 0),
        )

    def forward(self, x):
        x = self.net(x)
        if self.training:
            return x
        else:
            return torch.clamp(x, 0., 1.)


if __name__ == "__main__":
    import torch
    from nunif.models import create_model
    device = "cuda:0"
    model = create_model(VGG7.name, in_channels=3, out_channels=3).to(device)
    print(model)
    x = torch.zeros((1, 3, 256, 256)).to(device)
    with torch.no_grad():
        z = model(x)
        print(z.shape)
        print(model.name, model.i2i_offset, model.i2i_scale)
