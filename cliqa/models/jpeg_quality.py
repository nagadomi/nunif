import torch
import torch.nn as nn
from nunif.models import Model, register_model
from nunif.modules.res_block import ResBlockBNReLU


@register_model
class JPEGQuality(Model):
    name = "cliqa.jpeg_quality"

    def __init__(self):
        super().__init__({})
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            ResBlockBNReLU(128, 128),
            nn.MaxPool2d((2, 2)),
            ResBlockBNReLU(128, 128),
            nn.MaxPool2d((2, 2)),
        )
        self.quality_output = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )
        self.subsampling_output = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def preprocess(x):
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        y = r * 0.299 + g * 0.587 + b * 0.114
        cb = (b - y) * 0.564 + 0.5
        cr = (r - y) * 0.713 + 0.5

        x = torch.cat([y, cb, cr, r, g, b], dim=1)
        x = x * 2. - 1.
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.preprocess(x)
        x = self.features(x)
        quality = self.quality_output(x).view(B, -1)
        subsampling = self.subsampling_output(x).view(B, -1)
        return quality, subsampling


def _test():
    model = JPEGQuality().cuda()
    x = torch.zeros((4, 3, 128, 128)).cuda()
    print([z.shape for z in model(x)])


if __name__ == "__main__":
    _test()
