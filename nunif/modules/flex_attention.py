import torch
import torch.nn as nn
from .permute import bchw_to_bnc, bnc_to_bchw
from .init import basic_module_init
from torch import Tensor


try:
    from torch.nn.attention.flex_attention import flex_attention
except ModuleNotFoundError:
    flex_attention = None


def noop_score_mod(score, b, h, q_idx, kv_idx):
    return score


def sliced_flex_attention(q, k, v, num_heads, score_mod):
    B, N, C = q.shape  # batch, sequence, feature
    assert C % num_heads == 0
    qkv_dim = C // num_heads
    q = q.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    k = k.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    v = v.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    x = flex_attention(q, k, v, score_mod=score_mod)
    return x.permute(0, 2, 1, 3).reshape(B, N, qkv_dim * num_heads)


class FlexAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        assert flex_attention is not None, "torch version does not support flex_attention"

        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 3)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)
        basic_module_init(self)

    def forward(self, x, score_mod=noop_score_mod):
        q, k, v = self.qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_flex_attention(q, k, v, self.num_heads, score_mod=score_mod)
        x = self.head_proj(x)
        return x


class WindowNeighborhoodMHA2d(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, radius, mask=True, relative_bias=False, qkv_dim=None):
        assert mask or relative_bias
        super().__init__()

        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.register_buffer("max_distance", torch.tensor(radius + 0.5, dtype=torch.float32))
        # NOTE: should be int32. long is very slow
        self.register_buffer("window_h", torch.tensor(self.window_size[0], dtype=torch.int32))
        self.mha = FlexAttention(embed_dim, num_heads, qkv_dim=qkv_dim)
        self.relative_bias = relative_bias
        self.mask = mask

    def forward(self, x, layer_norm=None):
        # NOTE: should compile this function
        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)

        if self.relative_bias and self.mask:
            def score_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
                window_h = self.window_h
                q_h, q_w, k_h, k_w = q_idx // window_h, q_idx % window_h, kv_idx // window_h, kv_idx % window_h
                distance = ((q_h - k_h) ** 2 + (q_w - k_w) ** 2) ** 0.5

                # score bias (log distance, soft)
                score = score - (distance + 1.0).log()
                # circular neighborhood mask (distance threshold, hard)
                score = torch.where((distance <= self.max_distance), score, -float("inf"))

                return score
        elif self.mask:
            def score_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
                window_h = self.window_h
                q_h, q_w, k_h, k_w = q_idx // window_h, q_idx % window_h, kv_idx // window_h, kv_idx % window_h
                distance = ((q_h - k_h) ** 2 + (q_w - k_w) ** 2) ** 0.5
                score = torch.where((distance <= self.max_distance), score, -float("inf"))
                return score
        elif self.relative_bias:
            def score_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
                window_h = self.window_h
                q_h, q_w, k_h, k_w = q_idx // window_h, q_idx % window_h, kv_idx // window_h, kv_idx % window_h
                distance = ((q_h - k_h) ** 2 + (q_w - k_w) ** 2) ** 0.5
                score = score - (distance + 1.0).log()
                return score

        x = self.mha(x, score_mod=score_mod)
        x = bnc_to_bchw(x, out_shape, self.window_size)
        return x


def _test():
    import time
    B, C, H, W = 1, 128, 96, 96
    NUM_HEADS = 4
    NUM_LAYERS = 3
    WINDOW_SIZE = [48, 32, 96]
    RADIUS = 6.0

    model = nn.Sequential(
        nn.Conv2d(3, C, kernel_size=3, padding=1, bias=True),
        # layers
        nn.Sequential(*[
            WindowNeighborhoodMHA2d(
                C, num_heads=NUM_HEADS, window_size=WINDOW_SIZE[i], radius=RADIUS,
                mask=True, relative_bias=True)
            for i in range(NUM_LAYERS)]),
    ).cuda()
    model = torch.compile(model)
    x = torch.ones((B, 3, H, W), dtype=torch.float32).cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        # compile, warmup
        model(x)

        t = time.time()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        print(1.0 / ((time.time() - t) / 100), "FPS")


if __name__ == "__main__":
    _test()
