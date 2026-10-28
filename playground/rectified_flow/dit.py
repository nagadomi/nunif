import math

import torch
import torch.nn.functional as F
from torch import nn

from nunif.models import Model
from nunif.modules.attention import WindowMHA2dV2
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.glu import align_up, swiglu
from nunif.modules.init import basic_module_init
from nunif.modules.norm import RMSNorm1, RMSNorm2d
from nunif.modules.rope import RoPE2d

DOWNSAMPLING_FACTOR = 8


def time_embed(
    t: torch.Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    out = torch.cat((cos, sin), dim=-1).reshape(t.shape[0], -1, 1, 1)
    return out


class AdaLNParam(nn.Module):
    def __init__(self, time_dim, dim):
        super().__init__()
        self.lin = nn.Conv2d(time_dim, dim * 6, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, t):
        return self.lin(F.silu(t)).chunk(6, dim=1)


class SwiGLU2d(nn.Module):
    def __init__(self, in_channels, mlp_ratio=3):
        super().__init__()
        mid = align_up(in_channels, mlp_ratio * 0.5, 32)
        self.norm = RMSNorm2d(in_channels)
        self.w1 = nn.Conv2d(in_channels, mid * 2, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid, in_channels, kernel_size=3, stride=1, padding=1)
        basic_module_init(self)

    def forward(self, x, norm_shift, norm_scale):
        x = self.norm(x)
        x = (1 + norm_scale) * x + norm_shift
        x = self.w1(x)
        x = swiglu(x, dim=1)
        x = self.w2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, num_heads, time_dim, mlp_ratio=3, window_size=112 // DOWNSAMPLING_FACTOR):
        super(Block, self).__init__()
        self.norm1 = RMSNorm1(in_channels)
        self.rope = RoPE2d(in_channels // num_heads, size=window_size, norm_layer=RMSNorm1)
        self.mha = WindowMHA2dV2(in_channels, num_heads, window_size=window_size, shift=False, gate=True)
        self.mlp = SwiGLU2d(in_channels, mlp_ratio=mlp_ratio)
        self.modulation = AdaLNParam(time_dim, in_channels)

    def forward(self, x, time_emb):
        (mha_gate, mha_scale, mha_shift, mlp_gate, mlp_scale, mlp_shift) = self.modulation(time_emb)

        @torch.compiler.disable
        def apply_gated_residual(x, gate, block_out):
            # workaround for compile error (maybe pytorch bug)
            return x + gate * block_out

        x = apply_gated_residual(
            x, mha_gate, self.mha(x, layer_norm=self.norm1, rope=self.rope, norm_scale=mha_scale, norm_shift=mha_shift)
        )
        x = apply_gated_residual(x, mlp_gate, self.mlp(x, norm_scale=mlp_scale, norm_shift=mlp_shift))

        return x


class LastLayer(nn.Module):
    def __init__(self, in_channels, time_dim, out_channels, use_norm_path=True):
        super().__init__()
        if use_norm_path:
            self.norm = RMSNorm2d(in_channels)
        else:
            self.norm = nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(time_dim, in_channels * 2, kernel_size=1, stride=1, padding=0),
        )
        for m in self.modulation:
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        scale, shift = self.modulation(t).chunk(2, dim=1)
        x = (1 + scale) * self.norm(x) + shift
        x = self.proj(x)
        return x


class Blocks(nn.Module):
    def __init__(self, in_channels, num_heads, num_layers, time_dim, mlp_ratio=3):
        super(Blocks, self).__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    in_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    time_dim=time_dim,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, time_emb):
        for m in self.blocks:
            x = m(x, time_emb)
        return x


class DiT(Model):
    name = "playground.rectified_flow.dit"

    def __init__(self, dim=384, time_dim=128, use_norm_path=True):
        assert dim % 64 == 0
        super().__init__(locals())
        C = dim
        self.time_dim = time_dim
        self.proj_dim = C
        self.in_proj = nn.Conv2d(3 * DOWNSAMPLING_FACTOR**2, C, kernel_size=1, stride=1, padding=0)
        self.blocks = Blocks(C, num_heads=C // 64, num_layers=12, time_dim=self.time_dim)
        self.last_layer = LastLayer(C, time_dim, 3 * DOWNSAMPLING_FACTOR**2, use_norm_path=use_norm_path)
        self.time_mlp = nn.Sequential(
            nn.Conv2d(time_dim, time_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(time_dim, time_dim, kernel_size=1),
        )

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x, t):
        # v-pred
        time_emb = self.time_mlp(time_embed(t, self.time_dim).to(x.dtype))

        x = F.pixel_unshuffle(x, DOWNSAMPLING_FACTOR)
        x = self.in_proj(x)
        x = self.blocks(x, time_emb)
        x = self.last_layer(x, time_emb)
        x = F.pixel_shuffle(x, DOWNSAMPLING_FACTOR)

        return x


def _test_model():
    g = DiT()
    x = torch.zeros((4, 3, 112, 112))
    t = torch.zeros((4,))
    z = g(x, t)
    assert z.shape == (4, 3, 112, 112)


if __name__ == "__main__":
    _test_model()
