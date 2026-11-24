import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import nunif.modules.attention as attention
from nunif.modules.attention import sliced_sdp


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attn_mask: torch.Tensor | None,
    chunk: int = 4096,
) -> torch.Tensor:
    """fp32 MATH-backend reference, computed in batch chunks with the mask sliced per chunk."""
    B, N, C = q.shape
    D = C // num_heads
    batch_mask = attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[0] != 1
    outputs = []
    with sdpa_kernel([SDPBackend.MATH]):
        for i in range(0, B, chunk):
            qc, kc, vc = (t[i : i + chunk].float().view(-1, N, num_heads, D).permute(0, 2, 1, 3) for t in (q, k, v))
            mask = attn_mask[i : i + chunk] if batch_mask and attn_mask is not None else attn_mask
            if mask is not None and mask.dtype != torch.bool:
                mask = mask.float()
            x = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask)
            outputs.append(x.permute(0, 2, 1, 3).reshape(-1, N, C))
    return torch.cat(outputs, dim=0)


def _make_mask(kind: str, B: int, H: int, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if kind == "none":
        return None
    if kind == "broadcast":
        return torch.randn(H, N, N, device=device, dtype=dtype)
    if kind == "batch":
        mask = torch.rand(B, 1, N, N, device=device) > 0.3
        mask[..., 0] = True  # keep at least one visible key per query
        return mask
    raise ValueError(kind)


@pytest.mark.parametrize("mask_kind", ["none", "broadcast", "batch"])
def test_sliced_sdp_chunking_cpu(monkeypatch: pytest.MonkeyPatch, mask_kind: str) -> None:
    monkeypatch.setattr(attention, "SDPA_MAX_GRID_DIM", 4)
    torch.manual_seed(0)
    B, H, N, D = 11, 2, 16, 8
    q, k, v = (torch.randn(B, N, H * D) for _ in range(3))
    mask = _make_mask(mask_kind, B, H, N, q.device, q.dtype)

    out = sliced_sdp(q, k, v, H, attn_mask=mask)
    ref = _reference(q, k, v, H, mask, chunk=B)
    assert out.shape == (B, N, H * D)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
@pytest.mark.parametrize(
    ("B", "H", "mask_kind"),
    [
        (40000, 2, "none"),  # B * H > 65535 but each dimension is within the limit
        (40000, 2, "broadcast"),
        (70000, 2, "none"),  # B > 65535, chunked path
        (70000, 2, "broadcast"),
        (70000, 1, "batch"),
    ],
)
def test_sliced_sdp_large_batch_cuda(B: int, H: int, mask_kind: str) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    N, D = 64, 16
    q, k, v = (torch.randn(B, N, H * D, device=device, dtype=torch.float16) for _ in range(3))
    mask = _make_mask(mask_kind, B, H, N, device, torch.float16)

    out = sliced_sdp(q, k, v, H, attn_mask=mask)
    # Surfaces deferred launch errors (ROCm reports them on the next kernel launch).
    torch.cuda.synchronize()
    ref = _reference(q, k, v, H, mask)
    assert out.shape == (B, N, H * D)
    assert torch.isfinite(out).all()
    per_batch_err = (out.float() - ref).abs().amax(dim=(1, 2))
    assert per_batch_err.max().item() < 0.02, f"bad batches: {(per_batch_err >= 0.02).nonzero().flatten()[:10]}"
    del q, k, v, out, ref
    torch.cuda.empty_cache()
