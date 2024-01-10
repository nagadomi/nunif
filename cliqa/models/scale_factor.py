import torch
import torch.nn as nn
from nunif.models import Model, register_model
from nunif.modules.res_block import ResBlockBNReLU

# output :min 1.0, max 2.0


@register_model
class ScaleFactor(Model):
    name = "cliqa.scale_factor"

    def __init__(self):
        super().__init__({})
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            ResBlockBNReLU(128, 128),
            nn.MaxPool2d((2, 2)),
            ResBlockBNReLU(128, 128),
            nn.MaxPool2d((2, 2)),
        )
        self.scale_factor_output = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
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
        x = x * 2. - 1.
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.preprocess(x)
        x = self.features(x)
        scale_factor = self.scale_factor_output(x).view(B, -1)
        if self.training:
            return scale_factor
        else:
            return torch.clamp(scale_factor, 1.0, 2.0)


def _test():
    model = ScaleFactor().cuda()
    x = torch.zeros((4, 3, 128, 128)).cuda()
    z = model(x)
    print(z.shape)


if __name__ == "__main__":
    _test()
