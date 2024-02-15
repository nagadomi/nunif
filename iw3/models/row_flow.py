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
        self.sbs_output = False

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
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        if not self.sbs_output:
            grid = grid + delta
            grid = grid.permute(0, 2, 3, 1)
            z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
            z = F.pad(z, (-28, -28, -28, -28))
            if self.training:
                return z
            else:
                return torch.clamp(z, 0., 1.)
        else:
            # Generate LR
            grid_l = (grid + delta).permute(0, 2, 3, 1)
            grid_r = (grid - delta).permute(0, 2, 3, 1)
            z_l = F.grid_sample(rgb, grid_l, mode="bilinear", padding_mode="border", align_corners=True)
            z_r = F.grid_sample(rgb, grid_r, mode="bilinear", padding_mode="border", align_corners=True)
            # concat channels
            z = torch.cat([z_l, z_r], dim=1)
            z = F.pad(z, (-28, -28, -28, -28))
            if self.training:
                return z
            else:
                return torch.clamp(z, 0., 1.)


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    model = create_model(name).to(device).eval()
    x = torch.zeros((4, 8, 256, 256)).to(device)
    with torch.inference_mode():
        z, *_ = model(x)
        print(z.shape)
        print(model.name, model.i2i_offset, model.i2i_scale)

    # benchmark
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(100):
            z = model(x)
    print(time.time() - t)


if __name__ == "__main__":
    _bench("sbs.row_flow")
