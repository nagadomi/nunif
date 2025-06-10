import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model, register_model_factory
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
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

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class MLBW(I2IBaseModel):
    name = "sbs.mlbw"

    def __init__(self, num_layers=2, base_dim=32):
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
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lv2 = nn.Sequential(
            WABlock(C, (4, 4), shift=False, num_heads=self.num_layers),
            WABlock(C, (4, 4), shift=False, num_heads=self.num_layers),
        )
        self.lv1_out = nn.Sequential(
            ReplicationPad2dNaive((4, 4, 0, 0), detach=True),
            nn.Conv2d(C // pack, self.num_layers * 2, kernel_size=(1, 9), stride=1, padding=0),
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
        if False:  # self.training:
            pad_method = random.choice([0, 1, 2])
            if pad_method == 1:
                pad_w2 = pad_w1 + pad_w2
                pad_w1 = 0
                pad_h2 = pad_h1 + pad_h2
                pad_h1 = 0
            elif pad_method == 2:
                pad_w1 = pad_w1 + pad_w2
                pad_w2 = 0
                pad_h1 = pad_h1 + pad_h2
                pad_h2 = 0

        return pad_w1, pad_w2, pad_h1, pad_h2

    def _forward(self, x):
        input_height, input_width = x.shape[2:]
        pad_w1, pad_w2, pad_h1, pad_h2 = self._calc_pad(x)
        x = replication_pad2d_naive(x, (pad_w1, pad_w2, pad_h1, pad_h2), detach=True)
        x = x1 = self.lv1_in(x)
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.lv2(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        x = self.lv1_out(x + x1)
        x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2), mode="constant")
        delta, layer_weight = x.chunk(2, dim=1)

        # apply softmax step 1
        layer_weight = layer_weight.to(torch.float32)
        layer_weight = torch.exp(layer_weight - layer_weight.amax(dim=1, keepdim=True))

        return delta, layer_weight

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

    @conditional_compile(["NUNIF_TRAIN"])
    def _forward_default_composite(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence
        delta, layer_weight = self._forward(x)
        delta = delta.to(torch.float32)
        delta_scale = torch.tensor(1.0 / (x.shape[-1] // 2 - 1), dtype=x.dtype, device=x.device)

        # warp layer_weight with delta.detach()
        layer_weights = []
        for i in range(delta.shape[1]):
            w = self._warp(layer_weight[:, i:i + 1, :, :], grid, delta[:, i:i + 1, :, :].detach(), delta_scale)
            layer_weights.append(w)
        layer_weight = torch.cat(layer_weights, dim=1)
        # apply softmax step 2
        layer_weight = layer_weight / (layer_weight.sum(dim=1, keepdim=True) + 1e-6)

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
        z, grid, delta, layer_weight = self._forward_default_composite(x)
        if self.training:
            grid = (grid[:, 0:1, :, :] / delta_scale).detach()
            return z, grid + delta, layer_weight
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training

        # pre-warp layer_weight
        h, w = x.shape[-2], x.shape[-1]
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                        torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        grid = torch.stack((mesh_x, mesh_y), 2)
        grid = grid.permute(2, 0, 1)
        delta_scale = torch.tensor(1.0 / (w // 2 - 1), dtype=x.dtype, device=x.device)
        delta, layer_weight = self._forward(x)
        delta = delta.to(torch.float32)

        layer_weights = []
        for i in range(delta.shape[1]):
            w = self._warp(layer_weight[:, i:i + 1, :, :], grid, delta[:, i:i + 1, :, :], delta_scale)
            layer_weights.append(w)

        # apply softmax step 2
        layer_weight = torch.cat(layer_weights, dim=1)
        layer_weight = layer_weight / (layer_weight.sum(dim=1, keepdim=True) + 1e-6)

        return delta, layer_weight

    def forward(self, x):
        if not self.delta_output:
            return self._forward_default(x)
        else:
            return self._forward_delta_only(x)


register_model_factory(
    "sbs.mlbw_l2",
    lambda **kwargs: MLBW(num_layers=2, base_dim=32, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l3",
    lambda **kwargs: MLBW(num_layers=3, base_dim=32, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l4",
    lambda **kwargs: MLBW(num_layers=4, base_dim=32, **kwargs)
)
register_model_factory(
    "sbs.mlbw_l3_16",
    lambda **kwargs: MLBW(num_layers=3, base_dim=16, **kwargs)
)


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
    # 420 FPS on RTX3070Ti
    _bench("sbs.mlbw_l2")
    # 300 FPS on RTX3070Ti
    _bench("sbs.mlbw_l3")
    # 240 FPS on RTX3070Ti
    _bench("sbs.mlbw_l4")
