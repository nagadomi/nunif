from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_bias import (
    WindowDistanceScoreBias,
    WindowRelativeScoreBias,
    WindowScoreBias,
    WindowScoreBias3d,
)
from .init import basic_module_init
from .permute import (
    bcdhw_to_bnc,
    bchw_to_bnc,
    bhwc_to_bnc,
    bnc_to_bcdhw,
    bnc_to_bchw,
    bnc_to_bhwc,
    window_partition2d,
)
from .replication_pad2d import ReplicationPad2dNaive
from .rope import RoPE2d

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def use_flash_attention(flag):
        if flag:
            return nullcontext()
        else:
            return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:

    def use_flash_attention(flag):
        return torch.backends.cuda.sdp_kernel(enable_flash=flag, enable_math=True, enable_mem_efficient=flag)


class SEBlock(nn.Module):
    """from Squeeze-and-Excitation Networks"""

    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)
        basic_module_init(self)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)
        return x * z.expand(x.shape)


class SEBlockNHWC(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, in_channels // reduction, bias=bias)
        self.lin2 = nn.Linear(in_channels // reduction, in_channels, bias=bias)
        basic_module_init(self)

    def forward(self, x):
        z = x.mean(dim=[1, 2], keepdim=True)
        z = F.relu(self.lin1(z), inplace=True)
        z = torch.sigmoid(self.lin2(z))
        return x * z


def sliced_sdp(q, k, v, num_heads, attn_mask=None, rope=None, dropout_p=0.0, is_causal=False, num_kv_heads=None):
    B, QN, _ = q.shape  # batch, sequence, feature
    KN = k.shape[1]
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert q.shape[-1] % num_heads == 0
    assert k.shape[-1] % num_kv_heads == 0
    assert k.shape == v.shape
    assert num_heads % num_kv_heads == 0
    q_dim = q.shape[-1] // num_heads
    kv_dim = k.shape[-1] // num_kv_heads

    # B, H, N, C // H
    q = q.view(B, QN, num_heads, q_dim).permute(0, 2, 1, 3)
    k = k.view(B, KN, num_kv_heads, kv_dim).permute(0, 2, 1, 3)
    v = v.view(B, KN, num_kv_heads, kv_dim).permute(0, 2, 1, 3)

    if rope is not None:
        q, k = rope(q, k)

    use_flash = B <= 65535  # avoid CUDA error: invalid configuration argument.
    with use_flash_attention(use_flash):
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=num_heads != num_kv_heads,
        )
    # B, N, (H, C // H)
    return x.permute(0, 2, 1, 3).reshape(B, QN, q_dim * num_heads)


def pad_shift_mask_token(x, mask_token, window_size, shift=(True, True)):
    mask_token = mask_token.to(x.dtype)

    if shift[1]:
        B, C, H, W = x.shape
        pad_w = mask_token.expand(B, C, H, window_size[1] // 2)
        x = torch.cat((pad_w, x, pad_w), dim=3)
    if shift[0]:
        B, C, H, W = x.shape
        pad_h = mask_token.expand(B, C, window_size[0] // 2, W)
        x = torch.cat((pad_h, x, pad_h), dim=2)
    return x


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None, qkv_bias=True, num_kv_heads=None):
        super().__init__()
        # require torch >= 2.0 (recommend torch >= 2.1.2)
        # nn.MultiheadAttention also has a bug with float attn_mask, so PyTorch 2.1 is required anyway.
        assert hasattr(F, "scaled_dot_product_attention"), (
            "torch version does not support F.scaled_dot_product_attention"
        )

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_proj = nn.Linear(embed_dim, qkv_dim * num_heads + qkv_dim * num_kv_heads * 2, bias=qkv_bias)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)
        basic_module_init(self)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False, rope=None):
        # x.shape: batch, sequence, feature
        q, k, v = self.qkv_proj(x).split(
            (self.qkv_dim * self.num_heads, self.qkv_dim * self.num_kv_heads, self.qkv_dim * self.num_kv_heads), dim=-1
        )
        x = sliced_sdp(
            q,
            k,
            v,
            self.num_heads,
            attn_mask=attn_mask,
            rope=rope,
            dropout_p=dropout_p,
            is_causal=is_causal,
            num_kv_heads=self.num_kv_heads,
        )
        x = self.head_proj(x)
        return x


class WindowMHA2d(nn.Module):
    """WindowMHA
    BCHW input/output
    """

    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None, shift=False, shift_mask_token=False):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift)
        self.pad_h = self.pad_w = 0
        if self.shift[0] or self.shift[1]:
            if self.shift[0]:
                assert self.window_size[0] % 2 == 0
                self.pad_h = self.window_size[0] // 2
            if self.shift[1]:
                assert self.window_size[1] % 2 == 0
                self.pad_w = self.window_size[1] // 2
            if shift_mask_token:
                self.shift_mask_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
                nn.init.trunc_normal_(self.shift_mask_bias, 0, 0.01)

        if not hasattr(self, "shift_mask_bias"):
            self.shift_mask_bias = None

        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads=num_heads, qkv_dim=qkv_dim, qkv_bias=True)
        basic_module_init(self)

    def forward(self, x, attn_mask=None, layer_norm=None):
        if self.shift[0] or self.shift[1]:
            if self.shift_mask_bias is not None:
                x = pad_shift_mask_token(x, self.shift_mask_bias, self.window_size, self.shift)
            else:
                x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)
        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)
        if self.shift[0] or self.shift[1]:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))
        return x


def gen_padded_attention_mask_2d(
    B: int, H: int, W: int, window_h: int, window_w: int, pad_h: int, pad_w: int, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        mask_pad = torch.zeros((1, 1, H + pad_h * 2, W + pad_w * 2), device=device, dtype=torch.bool)
        mask_pad[:, :, pad_h : pad_h + H, pad_w : pad_w + W] = True

        # (num_windows, N)
        N = window_h * window_w
        win_mask = bchw_to_bnc(mask_pad, (window_h, window_w)).view(-1, N)

        # (num_windows, 1, N, N)
        attn_mask = win_mask.unsqueeze(1) & win_mask.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(1)

        # (B * num_windows, 1, N, N)
        num_windows = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * num_windows, 1, N, N)

    return attn_mask


class WindowMHA2dV2(nn.Module):
    """WindowMHA2d with RoPE2d
    BCHW input/output
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        window_size: int | tuple[int, int],
        shift: bool | tuple[bool, bool] | list[bool] = False,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift)
        self.pad_h = self.pad_w = 0
        if self.shift[0]:
            assert self.window_size[0] % 2 == 0
            self.pad_h = self.window_size[0] // 2
        if self.shift[1]:
            assert self.window_size[1] % 2 == 0
            self.pad_w = self.window_size[1] // 2

        self.mha = MHA(in_channels, num_heads, qkv_bias=False, num_kv_heads=num_kv_heads)

    def forward(
        self,
        x: torch.Tensor,
        layer_norm: nn.Module | None = None,
        rope: nn.Module | None = None,
        norm_shift: torch.Tensor | None = None,
        norm_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.window_size[0] == 0 and W % self.window_size[1] == 0

        needs_pad = self.pad_h > 0 or self.pad_w > 0
        if needs_pad:
            # NOTE: Flash Attention does not support non-null attn_mask.
            #       Efficient Attention does not support GQA.
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)
            attn_mask = gen_padded_attention_mask_2d(
                B,
                H,
                W,
                self.window_size[0],
                self.window_size[1],
                self.pad_h,
                self.pad_w,
                x.device,
            )
        else:
            attn_mask = None

        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        if norm_scale is not None and norm_shift is not None:
            B_bnc, N_bnc, C_bnc = x.shape
            num_windows = B_bnc // B
            x = x.view(B, num_windows, N_bnc, C_bnc)
            x = x * (1.0 + norm_scale.view(B, 1, 1, C)) + norm_shift.view(B, 1, 1, C)
            x = x.view(B_bnc, N_bnc, C)

        x = self.mha(x, attn_mask=attn_mask, rope=rope)
        x = bnc_to_bchw(x, out_shape, self.window_size)
        if needs_pad:
            x = x[:, :, self.pad_h : self.pad_h + H, self.pad_w : self.pad_w + W]

        return x


class WindowMHA2dCLV2(nn.Module):
    """WindowMHA2d with RoPE2d
    BHWC input/output
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        window_size: int | tuple[int, int],
        shift: bool | tuple[bool, bool] | list[bool] = False,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift)
        self.pad_h = self.pad_w = 0
        if self.shift[0]:
            assert self.window_size[0] % 2 == 0
            self.pad_h = self.window_size[0] // 2
        if self.shift[1]:
            assert self.window_size[1] % 2 == 0
            self.pad_w = self.window_size[1] // 2

        self.mha = MHA(in_channels, num_heads, qkv_bias=False, num_kv_heads=num_kv_heads)

    def forward(
        self,
        x: torch.Tensor,
        layer_norm: nn.Module | None = None,
        rope: nn.Module | None = None,
        norm_shift: torch.Tensor | None = None,
        norm_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H % self.window_size[0] == 0 and W % self.window_size[1] == 0

        needs_pad = self.pad_h > 0 or self.pad_w > 0
        if needs_pad:
            x = F.pad(x, (0, 0, self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)
            attn_mask = gen_padded_attention_mask_2d(
                B,
                H,
                W,
                self.window_size[0],
                self.window_size[1],
                self.pad_h,
                self.pad_w,
                x.device,
            )
        else:
            attn_mask = None

        out_shape = x.shape
        x = bhwc_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        if norm_scale is not None and norm_shift is not None:
            B_bnc, N_bnc, C_bnc = x.shape
            num_windows = B_bnc // B
            x = x.view(B, num_windows, N_bnc, C_bnc)
            x = x * (1.0 + norm_scale.view(B, 1, 1, C)) + norm_shift.view(B, 1, 1, C)
            x = x.view(B_bnc, N_bnc, C)

        x = self.mha(x, attn_mask=attn_mask, rope=rope)
        x = bnc_to_bhwc(x, out_shape, self.window_size)
        if needs_pad:
            x = x[:, self.pad_h : self.pad_h + H, self.pad_w : self.pad_w + W, :]

        return x


class WindowSpatialReductionMHA2d(nn.Module):
    # NOTE: slow when window_size < 16
    #       kernel_size=2 causes misalignment when calculating the distance between q_idx and kv_idx
    def __init__(self, in_channels, num_heads, window_size, kernel_size=3, reduction=2, qkv_dim=None):
        if reduction != 2:
            # TODO: kernel_size and padding for stride
            raise NotImplementedError()
        assert kernel_size in {2, 3}
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        if qkv_dim is None:
            assert in_channels % num_heads == 0
            qkv_dim = in_channels // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(in_channels, qkv_dim * num_heads)

        if kernel_size == 3:
            self.kv_pad = ReplicationPad2dNaive((1,) * 4, detach=True)
            self.kv_proj = nn.Conv2d(in_channels, qkv_dim * num_heads * 2, kernel_size=3, stride=2, padding=0)
        elif kernel_size == 2:
            self.kv_pad = nn.Identity()
            self.kv_proj = nn.Conv2d(in_channels, qkv_dim * num_heads * 2, kernel_size=2, stride=2, padding=0)

        self.head_proj = nn.Linear(qkv_dim * num_heads, in_channels)
        basic_module_init(self)

    def forward(self, x, attn_mask=None):
        src = x
        out_shape = src.shape

        # k, v
        x = window_partition2d(x, self.window_size)
        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)
        kv = self.kv_proj(self.kv_pad(x)).permute(0, 2, 3, 1).reshape(B * N, -1, self.qkv_dim * self.num_heads * 2)
        k, v = kv.contiguous().split(self.qkv_dim * self.num_heads, dim=-1)
        # q
        x = x.permute(0, 2, 3, 1).reshape(B * N, H * W, C)
        q = self.q_proj(x)

        # mha
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = self.head_proj(x)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        return x


class OverlapWindowMHA2dV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        window_size: int | tuple[int, int],
        bhwc=False,
    ) -> None:
        super().__init__()
        self.bhwc = bhwc
        # num_heads -> num_heads // 2
        assert num_heads >= 2 and num_heads % 2 == 0
        # in_channels -> in_channels // 2
        # head_dim = (in_channels // 2) // (num_heads // 2) = (in_channels // num_heads)
        self.num_heads = num_heads // 2
        self.qkv_dim = in_channels // num_heads
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.pad_h = self.pad_w = 0
        assert self.window_size[0] % 2 == 0
        self.pad_h = self.window_size[0] // 2
        assert self.window_size[1] % 2 == 0
        self.pad_w = self.window_size[1] // 2

        self.qkv_proj = nn.Linear(in_channels, self.qkv_dim * self.num_heads * 3 * 2, bias=False)
        self.head_proj = nn.Linear(in_channels, in_channels)
        basic_module_init(self.qkv_proj)
        basic_module_init(self.head_proj)

    def forward(
        self,
        x: torch.Tensor,
        layer_norm: nn.Module | None = None,
        rope: nn.Module | None = None,
        norm_shift: torch.Tensor | None = None,
        norm_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.bhwc:
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC

        B, H, W, C = x.shape
        assert H % self.window_size[0] == 0 and W % self.window_size[1] == 0

        if layer_norm is not None:
            x = layer_norm(x)
        if norm_scale is not None and norm_shift is not None:
            x = x * (1.0 + norm_scale.view(B, 1, 1, C)) + norm_shift.view(B, 1, 1, C)

        # (q1, k1, v1), (q2, k2, v2)
        x1, x2 = self.qkv_proj(x).chunk(2, dim=-1)
        x2 = F.pad(x2, (0, 0, self.pad_w, self.pad_w, self.pad_h, self.pad_h, 0, 0), mode="constant", value=0)

        x1_out_shape = (x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3] // 3)
        x2_out_shape = (x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3] // 3)

        x1 = bhwc_to_bnc(x1, self.window_size)
        x2 = bhwc_to_bnc(x2, self.window_size)
        x2_atten_mask = gen_padded_attention_mask_2d(
            B,
            H,
            W,
            self.window_size[0],
            self.window_size[1],
            self.pad_h,
            self.pad_w,
            x.device,
        )

        B1 = x1.shape[0]
        x1_atten_mask = torch.ones(
            (x1.shape[0], 1, x1.shape[1], x1.shape[1]), dtype=x2_atten_mask.dtype, device=x2_atten_mask.device
        )

        x = torch.cat((x1, x2), dim=0)
        attn_mask = torch.cat((x1_atten_mask, x2_atten_mask), dim=0)

        q, k, v = x.split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, rope=rope)
        x1 = x[:B1]
        x2 = x[B1:]
        x1 = bnc_to_bhwc(x1, x1_out_shape, self.window_size)
        x2 = bnc_to_bhwc(x2, x2_out_shape, self.window_size)
        x2 = x2[:, self.pad_h : self.pad_h + H, self.pad_w : self.pad_w + W, :]

        x = torch.cat((x1, x2), dim=-1)
        x = self.head_proj(x)

        if not self.bhwc:
            x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW

        return x


class WindowMHA3d(nn.Module):
    """3D WindowMHA
    BCDHW input/output
    """

    def __init__(self, in_channels, num_heads, window_size=(4, 4, 4), qkv_dim=None, shift=False, qkv_bias=True):
        super().__init__()
        self.window_size = (
            window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size, window_size)
        )
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift, shift)
        self.pad_h = self.pad_w = self.pad_d = 0
        if any(self.shift):
            if self.shift[0]:
                assert self.window_size[0] % 2 == 0
                self.pad_d = self.window_size[0] // 2
            if self.shift[1]:
                assert self.window_size[1] % 2 == 0
                self.pad_h = self.window_size[1] // 2
            if self.shift[2]:
                assert self.window_size[2] % 2 == 0
                self.pad_w = self.window_size[2] // 2

        if not hasattr(self, "shift_mask_bias"):
            self.shift_mask_bias = None

        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads=num_heads, qkv_dim=qkv_dim, qkv_bias=qkv_bias)
        basic_module_init(self)

    def forward(self, x, attn_mask=None, layer_norm=None):
        if any(self.shift):
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, 0, 0), mode="constant", value=0)
            x = F.pad(x, (0, 0, 0, 0, self.pad_d, self.pad_d), mode="reflect")

        out_shape = x.shape
        x = bcdhw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bcdhw(x, out_shape, self.window_size)
        if any(self.shift):
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h, -self.pad_d, -self.pad_d))
        return x


class CrossMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention"), (
            "torch version does not support F.scaled_dot_product_attention"
        )

        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, qkv_dim * num_heads)
        self.kv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 2)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)
        basic_module_init(self)

    def forward(self, q, kv, attn_mask=None, dropout_p=0.0, is_causal=False):
        assert q.shape == kv.shape
        q = self.q_proj(q)
        k, v = self.kv_proj(kv).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        x = self.head_proj(x)
        return x


class WindowCrossMHA2d(nn.Module):
    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.num_heads = num_heads
        self.mha = CrossMHA(in_channels, num_heads, qkv_dim)

    def forward(self, x1, x2, attn_mask=None, layer_norm1=None, layer_norm2=None):
        out_shape = x1.shape
        x1 = bchw_to_bnc(x1, self.window_size)
        x2 = bchw_to_bnc(x2, self.window_size)
        if layer_norm1 is not None:
            x1 = layer_norm1(x1)
        if layer_norm2 is not None:
            x2 = layer_norm2(x2)
        x = self.mha(x1, x2, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        return x


class GMLP(nn.Module):
    # gMLP
    def __init__(self, embed_dim, seq_len, mlp_ratio=1):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, int(embed_dim * mlp_ratio * 2))
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, kernel_size=1, stride=1, bias=True)
        self.proj_out = nn.Linear(int(embed_dim * mlp_ratio * 2) // 2, embed_dim)

        basic_module_init(self.proj_in)
        basic_module_init(self.proj_out)
        nn.init.uniform_(self.proj_spatial.weight, -1e-3 / embed_dim, 1e-3 / embed_dim)
        nn.init.constant_(self.proj_spatial.bias, 1.0)

    def forward(self, x, norm1=None, norm2=None):
        # B, N, C = x.shape
        shortcut = x
        if norm1 is not None:
            x = norm1(x)
        x = self.proj_in(x)
        x = F.gelu(x)

        u, v = x.chunk(2, dim=-1)
        if norm2 is not None:
            v = norm2(v)
        v = self.proj_spatial(v)
        x = u * v

        x = self.proj_out(x)
        x = x + shortcut

        return x


class WindowGMLP2d(nn.Module):
    """
    WindowGMLP2d
    BCHW input/output
    """

    def __init__(self, in_channels, window_size=(4, 4), mlp_ratio=2, shift=False, shift_mask_token=False):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.shift = shift
        if self.shift:
            assert self.window_size[0] % 2 == 0 and self.window_size[1] % 2 == 0
            self.pad_h = self.window_size[0] // 2
            self.pad_w = self.window_size[1] // 2
            if shift_mask_token:
                self.shift_mask_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
                nn.init.trunc_normal_(self.shift_mask_bias, 0, 0.01)

        if not hasattr(self, "shift_mask_bias"):
            self.shift_mask_bias = None

        self.seq_len = self.window_size[0] * self.window_size[1]
        self.gmlp = GMLP(in_channels, seq_len=self.seq_len, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1=None, norm2=None):
        if self.shift:
            if self.shift_mask_bias is not None:
                x = pad_shift_mask_token(x, self.shift_mask_bias, self.window_size)
            else:
                x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)

        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        if self.shift:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))

        return x


class WindowGMLP3d(nn.Module):
    """
    3D WindowGMLP
    BCDHW input/output
    """

    def __init__(self, in_channels, window_size=(4, 4, 4), mlp_ratio=2, shift=False):
        super().__init__()
        self.window_size = (
            window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size, window_size)
        )
        self.shift = shift if isinstance(shift, (tuple, list)) else (shift, shift, shift)
        self.pad_h = self.pad_w = self.pad_d = 0

        if any(self.shift):
            if self.shift[0]:
                assert self.window_size[0] % 2 == 0
                self.pad_d = self.window_size[0] // 2
            if self.shift[1]:
                assert self.window_size[1] % 2 == 0
                self.pad_h = self.window_size[1] // 2
            if self.shift[2]:
                assert self.window_size[2] % 2 == 0
                self.pad_w = self.window_size[2] // 2

        self.seq_len = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.gmlp = GMLP(in_channels, seq_len=self.seq_len, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1=None, norm2=None):
        if any(self.shift):
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, 0, 0), mode="constant", value=0)
            x = F.pad(x, (0, 0, 0, 0, self.pad_d, self.pad_d), mode="reflect")

        out_shape = x.shape
        x = bcdhw_to_bnc(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = bnc_to_bcdhw(x, out_shape, self.window_size)

        if any(self.shift):
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h, -self.pad_d, -self.pad_d))

        return x


def _bench_spatial_reduction():
    import time

    kernel_size = 2
    for window_size in (8, 12, 16, 24, 32, 48):
        x = torch.zeros((16, 64, 96, 96)).cuda()
        mha1 = WindowMHA2d(64, 4, window_size=window_size).cuda().eval()
        mha2 = (
            WindowSpatialReductionMHA2d(64, 4, window_size=window_size, kernel_size=kernel_size, reduction=2)
            .cuda()
            .eval()
        )
        mha1 = torch.compile(mha1)
        mha2 = torch.compile(mha2)

        t = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            assert mha1(x).shape == x.shape
            assert mha2(x).shape == x.shape
        torch.cuda.synchronize()

        t = time.time()
        for i in range(100):
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                mha1(x)
        torch.cuda.synchronize()
        fps1 = round(1.0 / ((time.time() - t) / 100), 2)

        t = time.time()
        for i in range(100):
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                mha2(x)
        torch.cuda.synchronize()
        fps2 = round(1.0 / ((time.time() - t) / 100), 2)

        print(f"window_size={window_size} WindowMHA2d={fps1}FPS WindowSpatialReductionMHA2d={fps2}FPS")

    """
    kernel_size=3
    window_size=8 WindowMHA2d=814.12FPS WindowSpatialReductionMHA2d=595.95FPS
    window_size=12 WindowMHA2d=654.66FPS WindowSpatialReductionMHA2d=634.88FPS
    window_size=16 WindowMHA2d=862.53FPS WindowSpatialReductionMHA2d=730.9FPS
    window_size=24 WindowMHA2d=501.31FPS WindowSpatialReductionMHA2d=618.97FPS
    window_size=32 WindowMHA2d=390.05FPS WindowSpatialReductionMHA2d=639.29FPS
    window_size=48 WindowMHA2d=205.52FPS WindowSpatialReductionMHA2d=438.88FPS

    kernel_size=2
    window_size=8 WindowMHA2d=812.55FPS WindowSpatialReductionMHA2d=749.31FPS
    window_size=12 WindowMHA2d=658.24FPS WindowSpatialReductionMHA2d=835.25FPS
    window_size=16 WindowMHA2d=864.63FPS WindowSpatialReductionMHA2d=1002.4FPS
    window_size=24 WindowMHA2d=503.62FPS WindowSpatialReductionMHA2d=783.26FPS
    window_size=32 WindowMHA2d=392.49FPS WindowSpatialReductionMHA2d=852.0FPS
    window_size=48 WindowMHA2d=206.99FPS WindowSpatialReductionMHA2d=528.88FPS
    """


def _test_neighborhood():
    from .flex_attention import WindowNeighborhoodMHA2d

    with torch.no_grad():
        x = torch.rand((1, 32, 32, 32))
        na1 = WindowNeighborhoodMHA2d(32, num_heads=4, window_size=(16, 16), max_distance=3.5, relative_bias=True)
        na2 = WindowMHA2d(32, num_heads=4, window_size=(16, 16))

        na2.mha.qkv_proj = na1.mha.qkv_proj
        na2.mha.head_proj = na1.mha.head_proj

        na2_score_mod = WindowDistanceScoreBias((16, 16), max_distance=3.5)

        z1 = na1(x)
        z2 = na2(x, attn_mask=na2_score_mod())
        diff = (z1 - z2).abs().sum()
        print(diff)
        assert diff < 1e-4


def _test_shift():
    mha1 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(False, True))
    mha2 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(True, False))
    mha3 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(False, False))
    mha4 = WindowMHA2d(32, num_heads=2, window_size=4, shift=True)
    mha5 = WindowMHA2d(32, num_heads=2, window_size=4, shift=False)

    x = torch.zeros((1, 32, 64, 64))
    assert mha1(x).shape == mha2(x).shape == mha3(x).shape == mha4(x).shape == mha5(x).shape

    mha1 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(False, True), shift_mask_token=True)
    mha2 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(True, False), shift_mask_token=True)
    mha3 = WindowMHA2d(32, num_heads=2, window_size=4, shift=(False, False), shift_mask_token=True)
    mha4 = WindowMHA2d(32, num_heads=2, window_size=4, shift=True, shift_mask_token=True)
    mha5 = WindowMHA2d(32, num_heads=2, window_size=4, shift=False, shift_mask_token=True)

    x = torch.zeros((1, 32, 64, 64))
    assert mha1(x).shape == mha2(x).shape == mha3(x).shape == mha4(x).shape == mha5(x).shape

    # I also tested with print debug at WindowMHA2d.forward


def _test_2d():
    x = torch.zeros((4, 32, 64, 64))
    window_size = (8, 8)
    bias = WindowScoreBias(window_size)
    mha = WindowMHA2d(32, num_heads=4, window_size=window_size)
    assert mha(x, attn_mask=bias()).shape == x.shape


def _test_3d():
    x = torch.zeros((4, 32, 16, 64, 64))
    window_size = (4, 8, 8)
    bias = WindowScoreBias3d(window_size)
    mha = WindowMHA3d(32, num_heads=4, window_size=window_size)
    assert mha(x, attn_mask=bias()).shape == x.shape


def _test_bias():
    mha = WindowMHA2d(64, 4, window_size=8).cuda().eval()
    x = torch.zeros((4, 64, 32, 32)).cuda()

    bias = WindowDistanceScoreBias(8, 8).cuda()
    mha(x, attn_mask=bias())
    bias = WindowDistanceScoreBias((8, 8), num_heads=4).cuda()
    mha(x, attn_mask=bias())
    bias = WindowDistanceScoreBias((8, 8), max_distance=3.5).cuda()
    mha(x, attn_mask=bias())
    bias = WindowDistanceScoreBias((8, 8), max_distance=3.5, num_heads=4).cuda()
    mha(x, attn_mask=bias())
    bias = WindowDistanceScoreBias((8, 8), max_distance=[3.5] * 4, num_heads=4).cuda()
    mha(x, attn_mask=bias())
    bias = WindowDistanceScoreBias((8, 8), max_distance=[3.5] * 2, num_heads=4).cuda()
    mha(x, attn_mask=bias())


def _test_bias2():
    bias = WindowRelativeScoreBias(window_size=3, num_heads=4)
    bias()
    bias = WindowRelativeScoreBias(window_size=3)
    bias()


def _test_2d_v2():
    dim = 64
    num_heads = 4
    window_size = (8, 8)

    # uses RoPE2d
    rope = RoPE2d(dim // num_heads, window_size, norm_layer=nn.LayerNorm).cuda()
    x = torch.zeros((4, dim, 32, 32)).cuda()
    for shift in [False, True, (True, False), (False, True)]:
        mha = WindowMHA2dV2(dim, num_heads=num_heads, window_size=window_size, shift=shift).cuda()
        assert mha(x, rope=rope).shape == x.shape


def _test_2d_cl_v2():
    dim = 64
    num_heads = 4
    window_size = (8, 8)
    x = torch.rand((4, dim, 32, 32)).cuda()
    x_cl = x.permute(0, 2, 3, 1).contiguous()

    for shift in [False, True, (True, False), (False, True)]:
        mha = WindowMHA2dV2(dim, num_heads=num_heads, window_size=window_size, shift=shift).cuda().eval()
        mha_cl = WindowMHA2dCLV2(dim, num_heads=num_heads, window_size=window_size, shift=shift).cuda().eval()

        # sync weights
        mha_cl.mha.load_state_dict(mha.mha.state_dict())

        with torch.inference_mode():
            y = mha(x)
            y_cl = mha_cl(x_cl)
            diff = (y - y_cl.permute(0, 3, 1, 2)).abs().max()
            # print(f"shift={shift} diff={diff}")
            assert diff < 1e-4


def _test_overlap_v2():
    dim = 64
    num_heads = 4
    window_size = (8, 8)
    x = torch.zeros((4, dim, 32, 32)).cuda()
    mha = OverlapWindowMHA2dV2(dim, num_heads=num_heads, window_size=window_size).cuda()
    mha(x)


def _test_gqa():
    dim = 64
    num_q_heads = 4
    num_kv_heads = 2
    window_size = (8, 8)

    mha = WindowMHA2dV2(dim, num_heads=num_q_heads, window_size=window_size, num_kv_heads=num_kv_heads).cuda()
    x = torch.zeros((4, dim, 32, 32)).cuda()
    mha(x)


def _test_gen_padded_attention_mask_2d():
    print("_test_gen_padded_attention_mask_2d")
    mask = gen_padded_attention_mask_2d(1, 4, 4, 2, 2, 1, 1, torch.device("cpu"))
    print(mask.shape)
    print(mask)


def _bench_gqa(do_compile=False):
    import time

    from torch.nn.attention import SDPBackend, sdpa_kernel

    N = 100
    IMG_SIZE = 256
    WINDOW_SIZE = 8
    B = (IMG_SIZE // WINDOW_SIZE) ** 2
    L = WINDOW_SIZE * WINDOW_SIZE
    dim = 256
    head_dim = 32
    num_heads = dim // head_dim

    print(f"\n**** _bench_gqa: compile={do_compile}")

    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ):
        print(f"** backend {backend}")
        torch.compiler.reset()
        mha = MHA(dim, num_heads=num_heads, qkv_bias=False).cuda().half()
        gqa = MHA(dim, num_heads=num_heads, num_kv_heads=num_heads // 2, qkv_bias=False).cuda().half()
        if do_compile:
            mha = torch.compile(mha)
            gqa = torch.compile(gqa)
        x = torch.rand((B, L, dim)).cuda().half()

        try:
            with sdpa_kernel([backend]):
                with torch.inference_mode():
                    mha(x)
                torch.cuda.synchronize()

                t = time.perf_counter()
                with torch.inference_mode():
                    for i in range(N):
                        mha(x)
                torch.cuda.synchronize()
                fps = round(1.0 / ((time.perf_counter() - t) / N), 3)
                print("MHA", fps)

                with torch.inference_mode():
                    gqa(x)
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    for i in range(N):
                        gqa(x)
                torch.cuda.synchronize()
                fps = round(1.0 / ((time.perf_counter() - t) / N), 3)
                print("GQA", fps)
        except RuntimeError:
            print("Error: skip")

    """
    **** _bench_gqa: compile=False
    ** backend SDPBackend.FLASH_ATTENTION
    MHA 1426.344
    GQA 1699.622
    ** backend SDPBackend.EFFICIENT_ATTENTION
    MHA 1343.229
    Error: skip
    ** backend SDPBackend.CUDNN_ATTENTION
    MHA 1657.916
    GQA 2078.301
    ** backend SDPBackend.MATH
    MHA 250.919
    GQA 271.549

    **** _bench_gqa: compile=True
    ** backend SDPBackend.FLASH_ATTENTION
    MHA 1427.617
    GQA 1731.205
    ** backend SDPBackend.EFFICIENT_ATTENTION
    MHA 1426.413
    Error: skip
    ** backend SDPBackend.CUDNN_ATTENTION
    MHA 1418.113
    GQA 1725.839
    ** backend SDPBackend.MATH
    MHA 1419.065
    GQA 1726.6
    """


if __name__ == "__main__":
    _bench_gqa(do_compile=False)
    _bench_gqa(do_compile=True)
    # _bench_spatial_reduction()
    # _test_gen_padded_attention_mask_2d()
    _test_gqa()
    _test_overlap_v2()
    _test_neighborhood()
    _test_shift()
    _test_bias()
    _test_bias2()
    _test_2d_v2()
    _test_2d_cl_v2()
    _test_2d()
    _test_3d()
    pass
