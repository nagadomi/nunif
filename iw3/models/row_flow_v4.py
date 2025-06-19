# 1 layer version of mlbw
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.replication_pad2d import (
    ReplicationPad2dNaive,
    replication_pad2d
)


OFFSET = 32


class WABlock(nn.Module):
    def __init__(self, in_channels, window_size, shift, num_heads):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=num_heads, window_size=window_size, shift=shift)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2dNaive((1, 1, 1, 1), detach=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
        )
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class RowFlowV4(I2IBaseModel):
    name = "sbs.row_flow_v4"

    def __init__(self):
        super(RowFlowV4, self).__init__(locals(), scale=1, offset=OFFSET, in_channels=8, blend_size=4)
        self.downscaling_factor = (1, 8)
        self.mod = 4
        pack = self.downscaling_factor[0] * self.downscaling_factor[1]
        C = 64
        assert C >= pack
        self.lv1_in = nn.Sequential(
            ReplicationPad2dNaive((4, 4, 0, 0), detach=True),
            nn.Conv2d(3, C // pack, kernel_size=(1, 9), stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lv2 = nn.Sequential(
            WABlock(C, (4, 4), shift=(False, True), num_heads=2),
            WABlock(C, (4, 4), shift=(False, False), num_heads=2),
        )
        self.lv1_out = nn.Sequential(
            ReplicationPad2dNaive((4, 4, 0, 0), detach=True),
            nn.Conv2d(C // pack, 1, kernel_size=(1, 9), stride=1, padding=0),
        )
        self.delta_output = False
        self.symmetric = False

    def _calc_pad(self, x):
        input_height, input_width = x.shape[2:]
        pad_w = (self.mod * self.downscaling_factor[1]) - input_width % (self.mod * self.downscaling_factor[1])
        pad_h = (self.mod * self.downscaling_factor[0]) - input_height % (self.mod * self.downscaling_factor[0])
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        return pad_w1, pad_w2, pad_h1, pad_h2

    def _forward(self, x):
        input_height, input_width = x.shape[2:]
        pad_w1, pad_w2, pad_h1, pad_h2 = self._calc_pad(x)
        x = replication_pad2d(x, (pad_w1, pad_w2, pad_h1, pad_h2))
        x = x1 = self.lv1_in(x)
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.lv2(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        x = self.lv1_out(x + x1)
        x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2), mode="constant")

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
        delta = delta.to(torch.float32)
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)
        z = self._warp(rgb, grid, delta, delta_scale)
        z = F.pad(z, (-OFFSET,) * 4)

        if self.training:
            return z, ((grid[:, 0:1, :, :] / delta_scale).detach() + delta)
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training
        delta = self._forward(x)
        delta = delta.to(torch.float32)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
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
    _bench("sbs.row_flow_v4")
