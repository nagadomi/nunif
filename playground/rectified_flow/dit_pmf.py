import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nunif.models import Model
from nunif.modules.compile_wrapper import conditional_compile

from .dit import DOWNSAMPLING_FACTOR, Blocks, LastLayer, time_embed


def _make_time_mlp(time_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(time_dim, time_dim, kernel_size=1),
        nn.SiLU(),
        nn.Conv2d(time_dim, time_dim, kernel_size=1),
    )


class DiTPMF(Model):
    name = "playground.rectified_flow.dit_pmf"

    def __init__(
        self,
        dim: int = 384,
        time_dim: int = 128,
        num_layers: int = 8,
        aux_head_depth: int = 4,
    ) -> None:
        assert dim % 64 == 0
        assert 0 < aux_head_depth < num_layers
        super().__init__(locals())
        C = dim
        num_heads = C // 64
        patch_channels = 3 * DOWNSAMPLING_FACTOR**2

        self.time_dim = time_dim
        self.proj_dim = C
        self.in_proj = nn.Conv2d(patch_channels, C, kernel_size=1, stride=1, padding=0)

        shared_depth = num_layers - aux_head_depth
        self.shared_blocks = Blocks(C, num_heads=num_heads, num_layers=shared_depth, time_dim=time_dim)
        self.u_heads = Blocks(C, num_heads=num_heads, num_layers=aux_head_depth, time_dim=time_dim)
        self.v_heads = Blocks(C, num_heads=num_heads, num_layers=aux_head_depth, time_dim=time_dim)

        self.u_last_layer = LastLayer(C, time_dim, patch_channels)
        self.v_last_layer = LastLayer(C, time_dim, patch_channels)
        self.t_mlp = _make_time_mlp(time_dim)
        self.h_mlp = _make_time_mlp(time_dim)

    def _encode_backbone(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use tfactor=1.0 so JVP derivatives d/dt and d/dh stay O(1) instead of scaling by 1000
        time_emb = self.t_mlp(time_embed(t, self.time_dim, tfactor=1.0).to(x.dtype)) + self.h_mlp(
            time_embed(h, self.time_dim, tfactor=1.0).to(x.dtype)
        )
        x = F.pixel_unshuffle(x, DOWNSAMPLING_FACTOR)
        x = self.in_proj(x)
        x = self.shared_blocks(x, time_emb)
        return x, time_emb

    @staticmethod
    def _run_head(
        x: torch.Tensor,
        time_emb: torch.Tensor,
        heads: Blocks,
        last_layer: nn.Module,
    ) -> torch.Tensor:
        x = heads(x, time_emb)
        x = last_layer(x, time_emb)
        return F.pixel_shuffle(x, DOWNSAMPLING_FACTOR)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x-prediction for both average (u) and instantaneous (v) heads
        feat, time_emb = self._encode_backbone(x, t, h)
        x_u = self._run_head(feat, time_emb, self.u_heads, self.u_last_layer)
        x_v = self._run_head(feat, time_emb, self.v_heads, self.v_last_layer)
        return x_u, x_v

    @conditional_compile(["NUNIF_TRAIN"])
    def forward_u(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        feat, time_emb = self._encode_backbone(x, t, h)
        return self._run_head(feat, time_emb, self.u_heads, self.u_last_layer)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward_v(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat, time_emb = self._encode_backbone(x, t, torch.zeros_like(t))
        return self._run_head(feat, time_emb, self.v_heads, self.v_last_layer)


def _test_model() -> None:
    from nunif.modules.attention import use_math_attention

    g = DiTPMF()
    x = torch.zeros((2, 3, 112, 112))
    t = torch.full((2,), 0.5)
    h = torch.full((2,), 0.25)
    x_u, x_v = g(x, t, h)
    assert x_u.shape == (2, 3, 112, 112)
    assert x_v.shape == (2, 3, 112, 112)
    assert g.forward_u(x, t, h).shape == (2, 3, 112, 112)
    assert g.forward_v(x, t).shape == (2, 3, 112, 112)

    if torch.cuda.is_available():
        g = g.cuda()
        x = torch.randn((2, 3, 112, 112), device="cuda")
        t = torch.full((2,), 0.7, device="cuda")
        r = torch.full((2,), 0.3, device="cuda")
        tx = torch.randn_like(x)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), use_math_attention(True):
            jvp_res: Any = torch.func.jvp(
                lambda z_in, t_in, r_in: g(z_in, t_in, t_in - r_in),
                (x, t, r),
                (tx, torch.ones_like(t), torch.zeros_like(r)),
                has_aux=True,
            )
            u, du_dt, v = jvp_res
            loss = ((u + du_dt.detach()).pow(2).mean() + v.pow(2).mean()) * math.sqrt(1.0)
        loss.backward()


if __name__ == "__main__":
    _test_model()
