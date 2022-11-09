import torch.nn as nn
from nunif.models import Model, register_model
from nunif.modules.inplace_clip import InplaceClip


class VGG7(Model):
    name = "waifu2x.vgg_7"

    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super(VGG7, self).__init__(VGG7.name, in_channels=in_channels,
                                   out_channels=out_channels, scale=1, offset=7)
        self.register_kwargs({"in_channels": in_channels, "out_channels": out_channels})
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
            InplaceClip(0, 1)
        )

    def forward(self, x):
        return self.net(x)


register_model(VGG7.name, VGG7)


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
        print(model.name, model.offset, model.scale)
