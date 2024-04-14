import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from nunif.models import I2IBaseModel, register_model
from nunif.modules.replication_pad2d import ReplicationPad2d


@register_model
class RowFlowV2(I2IBaseModel):
    name = "sbs.row_flow_v2"

    def __init__(self):
        super(RowFlowV2, self).__init__(locals(), scale=1, offset=28, in_channels=8, blend_size=4)
        self.feature = nn.Sequential(OrderedDict([
            ("pad0", ReplicationPad2d((1, 1, 0, 0))),
            ("0", nn.Conv2d(3, 16, kernel_size=(1, 3), stride=1, padding=0)),
            ("1", nn.ReLU(inplace=True))
        ]))
        self.non_overlap = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.overlap_residual = nn.Sequential(OrderedDict([
            ("pad0", ReplicationPad2d((4, 4, 0, 0))),
            ("0", nn.Conv2d(16, 16, kernel_size=(1, 9), stride=1, padding=0)),
            ("1", nn.ReLU(inplace=True)),
            ("pad1", ReplicationPad2d((4, 4, 0, 0))),
            ("2", nn.Conv2d(16, 32, kernel_size=(1, 9), stride=1, padding=0)),
            ("3", nn.ReLU(inplace=True)),
            ("pad2", ReplicationPad2d((4, 4, 0, 0))),
            ("4", nn.Conv2d(32, 32, kernel_size=(1, 9), stride=1, padding=0)),
            ("5", nn.ReLU(inplace=True)),
            ("pad3", ReplicationPad2d((1, 1, 1, 1))),
            ("6", nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0)),
        ]))
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.pre_pad = ReplicationPad2d((28,) * 4)
        self.delta_output = False

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

    def _warp(self, rgb, grid, delta, delta_scale):
        output_dtype = rgb.dtype
        rgb = rgb.to(torch.float32)
        grid = grid.to(torch.float32)
        delta = delta.to(torch.float32)
        delta_scale = delta_scale.to(torch.float32)

        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid + delta * delta_scale
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return z.to(output_dtype)

    def _forward_default(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        if self.training:
            delta1, delta2 = self._forward(x)
            z1 = self._warp(rgb, grid, delta1, self.delta_scale)
            z2 = self._warp(rgb, grid, delta2, self.delta_scale)
            z1 = F.pad(z1, (-28, -28, -28, -28))
            z2 = F.pad(z2, (-28, -28, -28, -28))
            return z2, z1, grid[:, 0:1, :, :] / self.delta_scale + delta2
        else:
            delta = self._forward(x)[1]
            z = self._warp(rgb, grid, delta, self.delta_scale)
            z = F.pad(z, (-28, -28, -28, -28))
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training
        x = self.pre_pad(x)
        delta = self._forward(x)[1]
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        delta = F.pad(delta, [-28] * 4)
        return delta

    def forward(self, x):
        if not self.delta_output:
            return self._forward_default(x)
        else:
            return self._forward_delta_only(x)


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    B = 4
    N = 100

    model = create_model(name).to(device).eval()
    x = torch.zeros((B, 8, 512, 512)).to(device)
    with torch.inference_mode():
        z, *_ = model(x)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params:,}")

    # benchmark
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    print(1 / ((time.time() - t)/(B * N)), "FPS")


if __name__ == "__main__":
    # 732 FPS on RTX3070Ti
    _bench("sbs.row_flow_v2")
