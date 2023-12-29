import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .permute import bchw_to_bnc, bnc_to_bchw


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


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # require torch >= 2.0 (recommend torch >= 2.1.2)
        # nn.MultiheadAttention also has a bug with float attn_mask, so PyTorch 2.1 is required anyway.
        assert hasattr(F, "scaled_dot_product_attention"), "torch version does not support F.scaled_dot_product_attention"

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.head_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False):
        B, N, C = x.shape  # batch, sequence, feature
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        # B, H, N, C // H
        q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=attn_mask, dropout_p=dropout_p,
                                           is_causal=is_causal)
        # B, N, (H, C // H)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.head_proj(x)
        return x


class WindowMHA2d(nn.Module):
    """ WindowMHA
    BCHW input/output
    """
    def __init__(self, in_channels, num_heads, window_size=(4, 4)):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads)

    def forward(self, x, attn_mask=None):
        src = x
        out_shape = src.shape
        x = bchw_to_bnc(x, self.window_size)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        return x


def _test():
    pass


if __name__ == "__main__":
    _test()
