import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model


@register_model
class RowFlow(I2IBaseModel):
    name = "sbs.row_flow"

    def __init__(self):
        # from diverdence==2.5, (0.5 * 2.5) / 100 * 2048 = 24, so offset must be > 24
        super(RowFlow, self).__init__(locals(), scale=1, offset=28, in_channels=8, blend_size=4)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(1, 3), stride=1, padding=(0, 1), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(1, 9), stride=1, padding=(0, 4), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 9), stride=1, padding=(0, 4), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 9), stride=1, padding=(0, 4), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        )
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        delta = self.conv(x) * self.delta_scale
        grid = grid + torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)

        z = F.pad(z, (-28, -28, -28, -28))
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


if __name__ == "__main__":
    device = "cuda:0"
    model = RowFlow().to(device)
    print(model)
    x = torch.zeros((1, 7, 2048, 2048)).to(device)
    with torch.no_grad():
        z = model(x)
        print(z.shape)
