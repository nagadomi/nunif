# DiT without timestep conditioning

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


class SwiGLU2d(nn.Module):
    def __init__(self, in_channels, mlp_ratio=3):
        super().__init__()
        mid = align_up(in_channels, mlp_ratio * 0.5, 32)
        self.norm = RMSNorm2d(in_channels)
        self.w1 = nn.Conv2d(in_channels, mid * 2, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid, in_channels, kernel_size=3, stride=1, padding=1)
        basic_module_init(self)

    def forward(self, x):
        x = self.norm(x)
        x = self.w1(x)
        x = swiglu(x, dim=1)
        x = self.w2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, num_heads, mlp_ratio=3, window_size=112 // DOWNSAMPLING_FACTOR):
        super(Block, self).__init__()
        self.norm1 = RMSNorm1(in_channels)
        self.rope = RoPE2d(in_channels // num_heads, size=window_size, norm_layer=RMSNorm1)
        self.mha = WindowMHA2dV2(in_channels, num_heads, window_size=window_size, shift=False, gate=True)
        self.mlp = SwiGLU2d(in_channels, mlp_ratio=mlp_ratio)

    def forward(self, x):

        @torch.compiler.disable
        def apply_gated_residual(x, block_out):
            # workaround for compile error (maybe pytorch bug)
            return x + block_out

        x = apply_gated_residual(x, self.mha(x, layer_norm=self.norm1, rope=self.rope))
        x = apply_gated_residual(x, self.mlp(x))

        return x


class LastLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm_path=True):
        super().__init__()
        if use_norm_path:
            self.norm = RMSNorm2d(in_channels)
        else:
            self.norm = nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x


class Blocks(nn.Module):
    def __init__(self, in_channels, num_heads, num_layers, mlp_ratio=3):
        super(Blocks, self).__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    in_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x


class DiTWithoutCond(Model):
    name = "playground.rectified_flow.dit_without_cond"

    def __init__(self, dim=384, use_norm_path=True):
        assert dim % 64 == 0
        super().__init__(locals())
        C = dim
        self.proj_dim = C
        self.in_proj = nn.Conv2d(3 * DOWNSAMPLING_FACTOR**2, C, kernel_size=1, stride=1, padding=0)
        self.blocks = Blocks(C, num_heads=C // 64, num_layers=12)
        self.last_layer = LastLayer(C, 3 * DOWNSAMPLING_FACTOR**2, use_norm_path=use_norm_path)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x, t=None):
        # v-pred
        x = F.pixel_unshuffle(x, DOWNSAMPLING_FACTOR)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.last_layer(x)
        x = F.pixel_shuffle(x, DOWNSAMPLING_FACTOR)

        return x


def _test_model():
    g = DiTWithoutCond()
    x = torch.zeros((4, 3, 112, 112))
    z = g(x)
    assert z.shape == (4, 3, 112, 112)


if __name__ == "__main__":
    _test_model()
