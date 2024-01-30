import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model


@register_model
class RowFlowV2(I2IBaseModel):
    name = "sbs.row_flow_v2"

    def __init__(self):
        super(RowFlowV2, self).__init__(locals(), scale=1, offset=28, in_channels=8, blend_size=4)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(1, 3), stride=1, padding=(0, 1), padding_mode="replicate"),
            nn.ReLU(inplace=True))
        self.non_overlap = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.overlap_residual = nn.Sequential(
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

    def _forward(self, x):
        x = self.feature(x)
        non_overlap = self.non_overlap(x)
        overlap_residual = self.overlap_residual(x)
        return non_overlap, non_overlap + overlap_residual

    def _warp(self, rgb, grid, delta):
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid + delta
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return z

    def forward(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        if self.training:
            delta1, delta2 = self._forward(x)
            delta1 = delta1 * self.delta_scale
            delta2 = delta2 * self.delta_scale
            z1 = self._warp(rgb, grid, delta1)
            z2 = self._warp(rgb, grid, delta2)
            z1 = F.pad(z1, (-28, -28, -28, -28))
            z2 = F.pad(z2, (-28, -28, -28, -28))
            return z2, z1, (grid[:, 0:1, :, :] + delta2) / self.delta_scale
        else:
            delta = self._forward(x)[1] * self.delta_scale
            if not self.sbs_output:
                z = self._warp(rgb, grid, delta)
                z = F.pad(z, (-28, -28, -28, -28))
                return torch.clamp(z, 0., 1.)
            else:
                z_l = self._warp(rgb, grid, delta)
                z_r = self._warp(rgb, grid, -delta)
                z = torch.cat([z_l, z_r], dim=1)
                z = F.pad(z, (-28, -28, -28, -28))
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
    _bench("sbs.row_flow_v2")
