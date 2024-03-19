import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .permute import bchw_to_bnc, bnc_to_bchw, bchw_to_bhwc, bhwc_to_bchw


class SEBlock(nn.Module):
    """ from Squeeze-and-Excitation Networks
    """
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        b, c, _, _ = x.size()
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

    def forward(self, x):
        B, H, W, C = x.size()
        z = x.mean(dim=[1, 2], keepdim=True)
        z = F.relu(self.lin1(z), inplace=True)
        z = torch.sigmoid(self.lin2(z))
        return x * z


class SNSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias))

    def forward(self, x):
        b, c, _, _ = x.size()
        z = F.adaptive_avg_pool2d(x, 1)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)
        return x * z.expand(x.shape)


def sliced_sdp(q, k, v, num_heads, attn_mask=None, dropout_p=0.0, is_causal=False):
    B, N, C = q.shape  # batch, sequence, feature
    assert C % num_heads == 0
    qkv_dim = C // num_heads
    # B, H, N, C // H
    q = q.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    k = k.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    v = v.view(B, N, num_heads, qkv_dim).permute(0, 2, 1, 3)
    x = F.scaled_dot_product_attention(q, k, v,
                                       attn_mask=attn_mask, dropout_p=dropout_p,
                                       is_causal=is_causal)
    # B, N, (H, C // H)
    return x.permute(0, 2, 1, 3).reshape(B, N, qkv_dim * num_heads)


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        # require torch >= 2.0 (recommend torch >= 2.1.2)
        # nn.MultiheadAttention also has a bug with float attn_mask, so PyTorch 2.1 is required anyway.
        assert hasattr(F, "scaled_dot_product_attention"), "torch version does not support F.scaled_dot_product_attention"

        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 3)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False):
        B, N, C = x.shape  # batch, sequence, feature
        q, k, v = self.qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        x = self.head_proj(x)
        return x


class WindowMHA2d(nn.Module):
    """ WindowMHA
    BCHW input/output
    """
    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads, qkv_dim)

    def forward(self, x, attn_mask=None, layer_norm=None):
        src = x
        out_shape = src.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        return x


class OverlapWindowMHA2d(nn.Module):
    # NOTE: Not much optimization
    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.pad_h = self.window_size[0] // 2
        self.pad_w = self.window_size[1] // 2
        self.num_heads = num_heads
        if qkv_dim is None:
            assert in_channels % num_heads == 0
            qkv_dim = in_channels // num_heads
        self.qkv_dim = qkv_dim
        self.qkv_proj1 = nn.Linear(in_channels, qkv_dim * num_heads * 3)
        self.qkv_proj2 = nn.Linear(in_channels, qkv_dim * num_heads * 3)
        self.head_proj = nn.Conv2d(qkv_dim * num_heads * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward_mha(self, x, qkv_proj, attn_mask=None):
        B, N, C = x.shape
        q, k, v = qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask)
        return x

    def forward(self, x, attn_mask=None, layer_norm=None):
        if layer_norm is not None:
            x = bhwc_to_bchw(layer_norm(bchw_to_bhwc(x)))
        x1 = x
        x2 = F.pad(x, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="constant", value=0)
        out_shape1 = x1.shape
        out_shape2 = x2.shape
        x1 = bchw_to_bnc(x1, self.window_size)
        x2 = bchw_to_bnc(x2, self.window_size)
        x1 = self.forward_mha(x1, self.qkv_proj1, attn_mask=attn_mask)
        x2 = self.forward_mha(x2, self.qkv_proj2, attn_mask=attn_mask)
        x1 = bnc_to_bchw(x1, out_shape1, self.window_size)
        x2 = bnc_to_bchw(x2, out_shape2, self.window_size)
        x2 = F.pad(x2, [-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h])
        x = torch.cat([x1, x2], dim=1)
        x = self.head_proj(x)

        return x


class CrossMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention"), "torch version does not support F.scaled_dot_product_attention"

        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, qkv_dim * num_heads)
        self.kv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 2)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)

    def forward(self, q, kv, attn_mask=None, dropout_p=0.0, is_causal=False):
        assert q.shape == kv.shape
        B, N, C = q.shape
        q = self.q_proj(q)
        k, v = self.kv_proj(kv).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        x = self.head_proj(x)
        return x


class WindowCrossMHA2d(nn.Module):
    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
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


if __name__ == "__main__":
    pass
