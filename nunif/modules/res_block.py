import torch
import torch.nn as nn
from .attention import SEBlock


"""
ResNet template

Typically, programmers want to removing code duplications,
 but I recommend that this file should be copied and then used.
"""


# TODO: test this


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        assert (stride in {1, 2})
        padding_mode = self.padding_mode()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, padding_mode=padding_mode,
                      bias=self.bias_enabled()),
            self.create_norm_layer(out_channels),
            self.create_activate_function(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, padding_mode=padding_mode,
                      bias=self.bias_enabled()),
            self.create_norm_layer(out_channels))
        if stride == 2 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                          bias=self.bias_enabled()),
                self.create_norm_layer(out_channels))
        else:
            self.identity = nn.Identity()

        self.attn = self.create_attention_layer(out_channels)
        self.act = self.create_activate_function()

    def forward(self, x):
        return self.attn(self.act(self.conv(x) + self.identity(x)))

    # factory methods

    def bias_enabled(self):
        return False

    def padding_mode(self):
        return "zeros"

    def create_activate_function(self):
        return nn.ReLU(inplace=True)

    def create_norm_layer(self, in_channels):
        return nn.BatchNorm2d(in_channels, momentum=0.01)

    def create_attention_layer(self, in_channels):
        return nn.Identity()


class ResGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, stride):
        super().__init__()
        assert (stride in {1, 2})
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(self.create_layer(in_channels, out_channels, stride=stride))
            else:
                layers.append(self.create_layer(out_channels, out_channels, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    # factory methods

    def create_layer(self, in_channels, out_channels, stride):
        return ResBlock(in_channels, out_channels, stride)


def _spec():
    device = "cuda:0"
    resnet = nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        ResGroup(64, 64, num_layers=3, stride=2),
        ResGroup(64, 128, num_layers=3, stride=2),
        ResGroup(128, 256, num_layers=3, stride=2),
        ResGroup(256, 512, num_layers=3, stride=2),
    ).to(device)
    x = torch.rand((8, 1, 256, 256)).to(device)
    z = resnet(x)
    print(resnet)
    print(z.shape)

    class CustomResBlock(ResBlock):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__(in_channels, out_channels, stride)

        def bias_enabled(self):
            return True

        def create_activate_function(self):
            return nn.LeakyReLU(0.1, inplace=True)

        def create_norm_layer(self, in_channels):
            return nn.Identity()

        def create_attention_layer(self, in_channels):
            return SEBlock(in_channels)

    class CustomResGroup(ResGroup):
        def __init__(self, in_channels, out_channels, num_layers, stride):
            super().__init__(in_channels, out_channels, num_layers, stride)

        def create_layer(self, in_channels, out_channels, stride):
            return CustomResBlock(in_channels, out_channels, stride)

    resnet = nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        CustomResGroup(64, 64, num_layers=3, stride=2),
        CustomResGroup(64, 64, num_layers=3, stride=2),
        CustomResGroup(64, 64, num_layers=3, stride=2),
        CustomResGroup(64, 64, num_layers=3, stride=2),
        CustomResGroup(64, 64, num_layers=3, stride=2),
    ).to(device)
    x = torch.rand((8, 1, 256, 256)).to(device)
    z = resnet(x)
    print(resnet)
    print(z.shape)


if __name__ == "__main__":
    _spec()
