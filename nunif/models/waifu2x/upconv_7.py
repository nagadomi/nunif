import torch.nn as nn
from .. model import Model
from ... modules.inplace_clip import InplaceClip


class UpConv7(Model):
    name = "waifu2x.upconv_7"

    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super(UpConv7, self).__init__(UpConv7.name, in_channels=in_channels, inner_scale=2, offset=14)
        self.register_kwargs({"in_channels": in_channels, "out_channels": out_channels})
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
            InplaceClip(0, 1)
        )

    def forward(self, x):
        return self.net(x)


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
        print(model.name, model.offset, model.inner_scale)
