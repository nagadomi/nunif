import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model, register_model_factory
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.replication_pad2d import (
    ReplicationPad2dNaive,
    replication_pad2d
)
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle


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

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class MLBW(I2IBaseModel):
    name = "sbs.mlbw"

    def __init__(self, num_layers=2, base_dim=32, small=False, cycle=False, hole_mask=False, **kwargs):
        super(MLBW, self).__init__(locals(), scale=1, offset=OFFSET, in_channels=8, blend_size=4)
        self.downscaling_factor = (1, 8)
        self.mod = 4
        pack = self.downscaling_factor[0] * self.downscaling_factor[1]
        self.num_layers = num_layers
        C = base_dim * self.num_layers
        assert C >= pack
        assert C // pack >= self.num_layers * 2
        self.lv1_in = nn.Sequential(
            ReplicationPad2dNaive((4, 4, 0, 0), detach=True),
            nn.Conv2d(3, C // pack, kernel_size=(1, 9), stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
        )
        if small:
            self.lv2 = nn.Sequential(
                WABlock(C, (4, 4), shift=(False, True), num_heads=self.num_layers),
                WABlock(C, (4, 4), shift=(False, False), num_heads=self.num_layers),
            )
        else:
            self.lv2 = nn.Sequential(
                WABlock(C, (4, 4), shift=(True, True), num_heads=self.num_layers),
                WABlock(C, (4, 4), shift=(False, False), num_heads=self.num_layers),
                WABlock(C, (4, 4), shift=(True, True), num_heads=self.num_layers),
                WABlock(C, (4, 4), shift=(False, False), num_heads=self.num_layers),
            )

        self.cycle = cycle
        self.hole_mask = hole_mask
        additional_output = 2 if self.hole_mask else 0
        self.lv1_out = nn.Sequential(
            ReplicationPad2dNaive((4, 4, 0, 0), detach=True),
            nn.Conv2d(C // pack, self.num_layers * 2 + additional_output, kernel_size=(1, 9), stride=1, padding=0),
        )
        self.delta_output = False
        self.symmetric = False

    def _calc_pad(self, x):
        input_height, input_width = x.shape[2:]
        pad_w = (self.mod * self.downscaling_factor[1]) - input_width % (self.mod * self.downscaling_factor[1])
        pad_h = (self.mod * self.downscaling_factor[0]) - input_height % (self.mod * self.downscaling_factor[0])
        if self.training:
            pad_w1 = random.randint(0, pad_w)
            pad_w2 = pad_w - pad_w1
            pad_h1 = random.randint(0, pad_h)
            pad_h2 = pad_h - pad_h1
        else:
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
        if self.hole_mask:
            delta, layer_weight = x[:, :self.num_layers * 2].chunk(2, dim=1)
            hole_mask_logits = x[:, self.num_layers * 2:]
        else:
            delta, layer_weight = x.chunk(2, dim=1)
            hole_mask_logits = None

        layer_weight = layer_weight.to(torch.float32)
        layer_weight = F.softmax(layer_weight, dim=1)

        return delta, layer_weight, hole_mask_logits

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

    def _forward_default_composite(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        delta, layer_weight, _ = self._forward(x)
        delta = delta.to(torch.float32)
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)

        # composite
        z = torch.zeros_like(rgb)
        for i in range(delta.shape[1]):
            d = delta[:, i:i + 1, :, :]
            w = layer_weight[:, i:i + 1, :, :]
            z = z + self._warp(rgb, grid, d, delta_scale) * w
        z = F.pad(z, (-OFFSET,) * 4)
        return z, grid, delta, layer_weight

    def _forward_default(self, x):
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)
        z, grid, delta, layer_weight, _ = self._forward_default_composite(x)
        if self.training:
            grid = (grid[:, 0:1, :, :] / delta_scale).detach()
            return z, grid + delta, layer_weight
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_hole_mask_composite(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        delta, layer_weight, hole_mask_logits = self._forward(x)
        delta = delta.to(torch.float32)
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)

        # composite
        z = torch.zeros_like(rgb)
        for i in range(delta.shape[1]):
            d = delta[:, i:i + 1, :, :]
            w = layer_weight[:, i:i + 1, :, :]
            z = z + self._warp(rgb, grid, d, delta_scale) * w
        z = F.pad(z, (-OFFSET,) * 4)
        hole_mask_logits = F.pad(hole_mask_logits, (-OFFSET,) * 4)
        return z, grid, delta, layer_weight, hole_mask_logits

    def _forward_hole_mask(self, x):
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)
        z, grid, delta, layer_weight, hole_mask_logits = self._forward_hole_mask_composite(x)
        if self.training:
            grid = (grid[:, 0:1, :, :] / delta_scale).detach()
            return z, grid + delta, layer_weight, hole_mask_logits
        else:
            hole_mask_logits = hole_mask_logits.to(torch.float32)
            return torch.clamp(z, 0., 1.), F.softmax(hole_mask_logits, dim=1)[:, 1:]

    def _forward_cycle_composite(self, x):
        rgb = x[:, 0:3, :, ]
        depth = x[:, 3:4, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        delta1, layer_weight1, _ = self._forward(x)
        delta1 = delta1.to(torch.float32)
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)

        # composite

        # warp
        warp_rgb = torch.zeros_like(rgb)
        warp_depth = torch.zeros_like(depth)
        for i in range(delta1.shape[1]):
            d = delta1[:, i:i + 1, :, :]
            w = layer_weight1[:, i:i + 1, :, :]
            warp_rgb = warp_rgb + self._warp(rgb, grid, d, delta_scale) * w
            warp_depth = warp_depth + self._warp(depth, grid, d, delta_scale) * w

        # reverse warp
        warp_depth = torch.flip(warp_depth, dims=[-1])
        warp_rgb_flip = torch.flip(warp_rgb, dims=[-1]).detach()
        x = torch.cat([warp_depth, x[:, 1:]], dim=1).detach()
        delta2, layer_weight2, _ = self._forward(x)
        delta2 = delta2.to(torch.float32)

        reverse_warp_rgb = torch.zeros_like(warp_rgb)
        for i in range(delta2.shape[1]):
            d = delta2[:, i:i + 1, :, :]
            w = layer_weight2[:, i:i + 1, :, :]
            reverse_warp_rgb = reverse_warp_rgb + self._warp(warp_rgb_flip, grid, d, delta_scale) * w

        reverse_warp_rgb = torch.flip(reverse_warp_rgb, dims=[-1])
        z1 = F.pad(warp_rgb, (-OFFSET,) * 4)
        z2 = F.pad(reverse_warp_rgb, (-OFFSET * 2,) * 4)
        src_rgb = F.pad(rgb.detach(), (-OFFSET * 2,) * 4)

        return z1, z2, grid, delta1, delta2, src_rgb

    def _forward_cycle(self, x):
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)
        z1, z2, grid, delta1, delta2, x = self._forward_cycle_composite(x)
        if self.training:
            grid = (grid[:, 0:1, :, :] / delta_scale).detach()
            return z1, z2, grid + delta1, grid + delta2, x
        else:
            return torch.clamp(z1, 0., 1.), torch.clamp(z2, 0., 1.), x

    def _forward_delta_only(self, x):
        assert not self.training
        delta, layer_weight, hole_mask_logits = self._forward(x)
        delta = delta.to(torch.float32)
        if self.hole_mask:
            hole_mask_logits = hole_mask_logits.to(torch.float32)
            return delta, layer_weight, F.softmax(hole_mask_logits, dim=1)[:, 1:]
        else:
            return delta, layer_weight

    def forward(self, x):
        if not self.delta_output:
            if self.cycle:
                return self._forward_cycle(x)
            elif self.hole_mask:
                return self._forward_hole_mask(x)
            else:
                return self._forward_default(x)
        else:
            return self._forward_delta_only(x)


register_model_factory(
    "sbs.mlbw_l2",
    lambda **kwargs: MLBW(num_layers=2, base_dim=32, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l4",
    lambda **kwargs: MLBW(num_layers=4, base_dim=32, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l2s",
    lambda **kwargs: MLBW(num_layers=2, base_dim=32, small=True, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l4s",
    lambda **kwargs: MLBW(num_layers=4, base_dim=32, small=True, **kwargs)
)

register_model_factory(
    "sbs.cycle_mlbw_l2",
    lambda **kwargs: MLBW(num_layers=2, base_dim=32, cycle=True, hole_mask=True, **kwargs)
)
register_model_factory(
    "sbs.mask_mlbw_l2",
    lambda **kwargs: MLBW(num_layers=2, base_dim=32, hole_mask=True, **kwargs)
)


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    B = 4
    N = 100

    model = create_model(name).to(device).eval()
    x = torch.zeros((B, 8, 512, 512)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")


if __name__ == "__main__":
    # 305 FPS on RTX3070Ti
    _bench("sbs.mlbw_l2")
    # 150 FPS on RTX3070Ti
    _bench("sbs.mlbw_l4")

    # 490 FPS on RTX3070Ti
    _bench("sbs.mlbw_l2s")
    # 250 FPS on RTX3070Ti
    _bench("sbs.mlbw_l4s")
