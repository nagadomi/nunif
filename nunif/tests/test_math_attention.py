import threading

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nunif.modules.attention import (
    MHA,
    WindowMHA2dV2,
    get_math_attention,
    set_math_attention,
    sliced_sdp,
    use_math_attention,
)
from nunif.modules.norm import RMSNorm1
from nunif.modules.rope import RoPE2d


def test_math_attention_flag_default_and_context() -> None:
    set_math_attention(False)
    assert get_math_attention() is False

    set_math_attention(True)
    assert get_math_attention() is True

    with use_math_attention(False):
        assert get_math_attention() is False
    assert get_math_attention() is True

    set_math_attention(False)
    assert get_math_attention() is False


def test_math_attention_thread_local() -> None:
    set_math_attention(False)

    other_thread_states: list[bool] = []
    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        other_thread_states.append(get_math_attention())
        with use_math_attention(True):
            other_thread_states.append(get_math_attention())
            entered.set()
            release.wait(timeout=5.0)
        other_thread_states.append(get_math_attention())

    t = threading.Thread(target=worker)
    t.start()
    try:
        assert entered.wait(timeout=5.0)
        assert get_math_attention() is False
    finally:
        release.set()
        t.join(timeout=5.0)

    assert other_thread_states == [False, True, False]
    assert get_math_attention() is False


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("num_kv_heads", [4, 2])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sliced_sdp_math_attention_jvp_matches_sdpa(
    device: torch.device,
    num_kv_heads: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(0)
    B, N, D = 2, 16, 128
    num_heads = 4
    kv_dim = (D // num_heads) * num_kv_heads
    q = torch.randn(B, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, kv_dim, device=device, dtype=torch.float32)
    v = torch.randn(B, N, kv_dim, device=device, dtype=torch.float32)
    tq = torch.randn_like(q)
    tk = torch.randn_like(k)
    tv = torch.randn_like(v)

    def fn(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        return sliced_sdp(q_in, k_in, v_in, num_heads=num_heads, num_kv_heads=num_kv_heads, is_causal=is_causal)

    set_math_attention(False)
    with sdpa_kernel([SDPBackend.MATH]):
        res_ref = torch.func.jvp(fn, (q, k, v), (tq, tk, tv))
    out_ref, jvp_ref = res_ref[0], res_ref[1]

    with use_math_attention(True):
        res_math = torch.func.jvp(fn, (q, k, v), (tq, tk, tv))
    out_math, jvp_math = res_math[0], res_math[1]

    assert get_math_attention() is False
    torch.testing.assert_close(out_math, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jvp_math, jvp_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_sliced_sdp_math_attention_masks(device: torch.device) -> None:
    torch.manual_seed(0)
    B, N, D = 2, 12, 64
    num_heads = 4
    q = torch.randn(B, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, D, device=device, dtype=torch.float32)

    bool_mask = torch.ones((B, 1, N, N), device=device, dtype=torch.bool)
    bool_mask[:, :, :, -2:] = False

    float_mask = torch.zeros((B, 1, N, N), device=device, dtype=torch.float32)
    float_mask[:, :, :, -2:] = -1e4

    for mask in (bool_mask, float_mask):
        set_math_attention(False)
        with sdpa_kernel([SDPBackend.MATH]):
            ref = sliced_sdp(q, k, v, num_heads=num_heads, attn_mask=mask)
        with use_math_attention(True):
            out = sliced_sdp(q, k, v, num_heads=num_heads, attn_mask=mask)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_mha_and_window_mha2d_v2_math_attention_jvp_and_backward(device: torch.device) -> None:
    torch.manual_seed(0)
    mha = MHA(embed_dim=128, num_heads=4, gate=True).to(device)
    x = torch.randn(2, 16, 128, device=device, dtype=torch.float32, requires_grad=True)
    tx = torch.randn_like(x)

    with use_math_attention(True):
        res = torch.func.jvp(lambda inp: mha(inp), (x,), (tx,))
        out, dout = res[0], res[1]
        loss = (out + dout).pow(2).mean()
        loss.backward()

    assert x.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(dout).all()
    assert torch.isfinite(x.grad).all()

    window_size = 8
    wmha = WindowMHA2dV2(in_channels=128, num_heads=4, window_size=window_size, shift=False, gate=True).to(device)
    rope = RoPE2d(128 // 4, size=window_size, norm_layer=RMSNorm1).to(device)
    img = torch.randn(2, 128, window_size, window_size, device=device, dtype=torch.float32, requires_grad=True)
    timg = torch.randn_like(img)

    with use_math_attention(True):
        w_res = torch.func.jvp(lambda inp: wmha(inp, rope=rope), (img,), (timg,))
        w_out, w_dout = w_res[0], w_res[1]
        w_loss = (w_out + w_dout).pow(2).mean()
        w_loss.backward()

    assert img.grad is not None
    assert torch.isfinite(w_out).all()
    assert torch.isfinite(w_dout).all()
    assert torch.isfinite(img.grad).all()
