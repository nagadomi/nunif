from collections.abc import Callable

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    i1 = (Ellipsis, slice(None, half))
    i2 = (Ellipsis, slice(half, None))
    if torch.onnx.is_in_onnx_export():
        return x * cos + torch.cat((-x[i2], x[i1]), dim=-1) * sin
    else:
        # This requires Torch 2.13 or later.
        sin = sin[i1]  # sin[i1] == sin[i2]
        # float32
        out = x * cos
        out[i1].addcmul_(x[i2], sin, value=-1)
        out[i2].addcmul_(x[i1], sin, value=1)
        return out


class RoPE2d(nn.Module):
    cos_h: torch.Tensor
    sin_h: torch.Tensor
    cos_w: torch.Tensor
    sin_w: torch.Tensor
    q_norm: nn.Module
    k_norm: nn.Module

    def __init__(
        self,
        head_dim: int,
        size: int | tuple[int, int],
        base: float = 10000.0,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be a multiple of 4"

        if isinstance(size, int):
            height = width = size
        else:
            height, width = size

        self.head_dim = head_dim
        self.height = height
        self.width = width
        self.dim_chunk = head_dim // 2

        cos_h, sin_h, cos_w, sin_w = self._precompute_freqs(base)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

        if norm_layer is not None:
            self.q_norm = norm_layer(head_dim)
            self.k_norm = norm_layer(head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    @torch.no_grad()
    def _precompute_freqs(self, base: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_h, sin_h = self._get_1d_sin_cos(base, self.height, self.dim_chunk)
        cos_w, sin_w = self._get_1d_sin_cos(base, self.width, self.dim_chunk)

        cos_h = cos_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        sin_h = sin_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        cos_w = cos_w.reshape(1, 1, 1, self.width, self.dim_chunk)
        sin_w = sin_w.reshape(1, 1, 1, self.width, self.dim_chunk)

        return cos_h, sin_h, cos_w, sin_w

    def _get_1d_sin_cos(self, base: float, max_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        exp = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base**exp)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, num_heads, N, head_dim = x.shape
        x_h = x[..., : self.dim_chunk].reshape(B, num_heads, self.height, self.width, self.dim_chunk)
        x_w = x[..., self.dim_chunk :].reshape(B, num_heads, self.height, self.width, self.dim_chunk)

        out_h = apply_rope(x_h, self.cos_h.to(x.dtype), self.sin_h.to(x.dtype))
        out_w = apply_rope(x_w, self.cos_w.to(x.dtype), self.sin_w.to(x.dtype))

        out_h = out_h.reshape(B, num_heads, N, self.dim_chunk)
        out_w = out_w.reshape(B, num_heads, N, self.dim_chunk)
        return torch.cat([out_h, out_w], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: [B, num_heads, N, head_dim] (N = h * w)
        """
        q = self.apply_rope(self.q_norm(q).to(q.dtype))
        k = self.apply_rope(self.k_norm(k).to(k.dtype))
        return q, k


def _bench(do_compile):
    import time

    device = "cuda:0"
    N = 20
    LAYERS = 10
    B = 64
    S = (128, 128)
    dim = 256
    num_heads = 4
    head_dim = dim // 4

    rope = RoPE2d(head_dim, S, norm_layer=nn.LayerNorm).cuda()
    x = torch.rand((B, num_heads, S[0] * S[1], head_dim), dtype=torch.float16, device="cuda")
    if do_compile:
        rope = torch.compile(rope)

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, _ = rope(x, x)
        print(z.shape, z.dtype)

    # check backward works
    with torch.autocast(device_type="cuda"):
        sum(
            rope(
                x + nn.Parameter(torch.zeros(1, dtype=x.dtype)).cuda(),
                x + nn.Parameter(torch.zeros(1, dtype=x.dtype)).cuda(),
            )
        ).sum().backward()

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            for _ in range(LAYERS):
                rope(x, x)
    torch.cuda.synchronize()
    print(round(1 / ((time.time() - t) / (B * N)), 3), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


def _visualize():

    import torchvision.transforms.functional as TF
    from torchvision.utils import make_grid

    from .permute import bchw_to_bnc, bnc_to_bchw

    C = 32
    IMG_SIZE = 32
    WINDOW = 8
    x = torch.ones((1, C, IMG_SIZE, IMG_SIZE))
    out_shape = x.shape
    rope = RoPE2d(C, WINDOW, base=10000)
    x = bchw_to_bnc(x, WINDOW)
    x = x.unsqueeze(1)
    x = rope.apply_rope(x)
    x = x.squeeze(1)
    x = bnc_to_bchw(x, out_shape, WINDOW)

    images = []
    for i in range(C):
        img = x[0, i : (i + 1)]
        img = (img + 2**0.5) / (2**0.5 * 2)
        images.append(img)

    img = make_grid(images, nrow=C // 2, padding=2, normalize=False)
    TF.to_pil_image(img).show()


if __name__ == "__main__":
    _bench(do_compile=False)
    _bench(do_compile=True)
    # _visualize()
