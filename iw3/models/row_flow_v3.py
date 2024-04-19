import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2d


OFFSET = 32


class WABlock(nn.Module):
    def __init__(self, in_channels, window_size, layer_norm=False):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=2, window_size=window_size)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.1, inplace=True))
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class RowFlowV3(I2IBaseModel):
    name = "sbs.row_flow_v3"

    def __init__(self):
        super(RowFlowV3, self).__init__(locals(), scale=1, offset=OFFSET, in_channels=8, blend_size=4)
        self.downscaling_factor = (1, 8)
        self.mod = 4 * 3
        pack = self.downscaling_factor[0] * self.downscaling_factor[1]
        C = 64
        assert C >= pack
        self.blocks = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            WABlock(C, (4, 4)),
            WABlock(C, (3, 3)),
        )
        self.last_layer = nn.Sequential(
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(C // pack, 1, kernel_size=3, stride=1, padding=0)
        )
        self.pre_pad = ReplicationPad2d((OFFSET,) * 4)
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.delta_output = False
        self.symmetric = False

    def _forward(self, x):
        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor[1]) - input_width % (self.mod * self.downscaling_factor[1])
        pad2 = (self.mod * self.downscaling_factor[0]) - input_height % (self.mod * self.downscaling_factor[0])
        x = replication_pad2d_naive(x, (0, pad1, 0, pad2))
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.blocks(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        x = F.pad(x, (0, -pad1, 0, -pad2), mode="constant")
        x = self.last_layer(x)
        return x

    def _warp(self, rgb, grid, delta, delta_scale):
        output_dtye = rgb.dtype
        rgb = rgb.to(torch.float32)
        grid = grid.to(torch.float32)
        delta = delta.to(torch.float32)
        delta_scale = delta_scale.to(torch.float32)

        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid + delta * delta_scale
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return z.to(output_dtye)

    def _forward_default(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence

        delta = self._forward(x)
        if self.symmetric:
            left = self._warp(rgb, grid, delta, self.delta_scale)
            right = self._warp(rgb, grid, -delta, self.delta_scale)
            left = F.pad(left, (-OFFSET,) * 4)
            right = F.pad(right, (-OFFSET,) * 4)
            z = torch.cat([left, right], dim=1)
        else:
            z = self._warp(rgb, grid, delta, self.delta_scale)
            z = F.pad(z, (-OFFSET,) * 4)

        if self.training:
            return z, ((grid[:, 0:1, :, :] / self.delta_scale).detach() + delta)
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training
        # TODO: maybe no need this padding
        x = self.pre_pad(x)
        delta = self._forward(x)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        delta = F.pad(delta, [-OFFSET] * 4)
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
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")


if __name__ == "__main__":
    # 540 FPS on RTX3070Ti
    _bench("sbs.row_flow_v3")
