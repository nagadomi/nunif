import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .permute import bchw_to_bnc, bnc_to_bchw, bchw_to_bhwc, bhwc_to_bchw, window_partition2d
from .init import basic_module_init
from .replication_pad2d import ReplicationPad2dNaive


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def use_flash_attention(flag):
        if flag:
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        else:
            return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:
    def use_flash_attention(flag):
        return torch.backends.cuda.sdp_kernel(enable_flash=flag, enable_math=True, enable_mem_efficient=flag)


class SEBlock(nn.Module):
    """ from Squeeze-and-Excitation Networks
    """
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


class SNSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias))
        basic_module_init(self)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1)
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)
        z = torch.sigmoid(z)
        return x * z.expand(x.shape)


def sliced_sdp(q, k, v, num_heads, attn_mask=None, dropout_p=0.0, is_causal=False):
    B, QN, C = q.shape  # batch, sequence, feature
    KN = k.shape[1]
    assert C % num_heads == 0
    qkv_dim = C // num_heads
    # B, H, N, C // H
    q = q.view(B, QN, num_heads, qkv_dim).permute(0, 2, 1, 3)
    k = k.view(B, KN, num_heads, qkv_dim).permute(0, 2, 1, 3)
    v = v.view(B, KN, num_heads, qkv_dim).permute(0, 2, 1, 3)

    use_flash = B <= 65535  # avoid CUDA error: invalid configuration argument.
    with use_flash_attention(use_flash):
        x = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=attn_mask, dropout_p=dropout_p,
                                           is_causal=is_causal)
    # B, N, (H, C // H)
    return x.permute(0, 2, 1, 3).reshape(B, QN, qkv_dim * num_heads)


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
        basic_module_init(self)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False):
        # x.shape: batch, sequence, feature
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
        basic_module_init(self)

    def forward(self, x, attn_mask=None, layer_norm=None):
        src = x
        out_shape = src.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)

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
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
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
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=0., is_causal=False)
        x = self.head_proj(x)
        x = bnc_to_bchw(x, out_shape, self.window_size)

        return x


class OverlapWindowMHA2d(nn.Module):
    # NOTE: Not much optimization. Not used.
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
        self.qkv_proj = nn.Conv2d(in_channels, qkv_dim * num_heads * 3, kernel_size=1, stride=1, padding=0)
        self.head_proj = nn.Conv2d(qkv_dim * num_heads, in_channels, kernel_size=1, stride=1, padding=0)

    def forward_mha(self, x, attn_mask=None):
        q, k, v = x.split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask)
        return x

    def forward(self, x, attn_mask=None, layer_norm=None):
        if layer_norm is not None:
            x = bhwc_to_bchw(layer_norm(bchw_to_bhwc(x)))
        x = self.qkv_proj(x)
        x1 = x
        x2 = F.pad(x, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="constant", value=0)
        out_shape1 = x1.shape
        out_shape2 = x2.shape
        x1 = bchw_to_bnc(x1, self.window_size)
        x2 = bchw_to_bnc(x2, self.window_size)
        x1 = self.forward_mha(x1, attn_mask=attn_mask)
        x2 = self.forward_mha(x2, attn_mask=attn_mask)
        x1 = bnc_to_bchw(x1, (out_shape1[0], x1.shape[-1], *out_shape1[2:]), self.window_size)
        x2 = bnc_to_bchw(x2, (out_shape2[0], x2.shape[-1], *out_shape2[2:]), self.window_size)
        x2 = F.pad(x2, [-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h])
        x = self.head_proj(x1 + x2)

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


@torch.no_grad()
def _gen_window_score_bias_input(window_size1, window_size2, reduction):
    N1 = window_size1[0] * window_size1[1]
    N2 = window_size2[0] * window_size2[1]

    positions1 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size1[0]),
                       torch.arange(0, window_size1[1]), indexing="ij"), dim=2).reshape(N1, 2)

    positions2 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size2[0]),
                       torch.arange(0, window_size2[1]), indexing="ij"), dim=2).reshape(N2, 2)
    positions2.mul_(reduction)

    delta = torch.zeros((N1, N2, 2), dtype=torch.long)
    for i in range(N1):
        for j in range(N2):
            delta[i][j] = positions1[i] - positions2[j]

    delta = delta.view(N1 * N2, 2)
    delta = [tuple(p) for p in delta.tolist()]
    unique_delta = sorted(list(set(delta)))
    index = [unique_delta.index(d) for d in delta]
    index = torch.tensor(index, dtype=torch.int64)
    unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
    unique_delta = unique_delta / unique_delta.abs().max()
    return index, unique_delta


class WindowScoreBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads

        index, unique_delta = _gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)
        if hidden_dim is None:
            hidden_dim = int((self.window_size1[0] * self.window_size1[1]) ** 0.5) * 2
        if self.num_heads is None:
            output_dim = 1
        else:
            output_dim = num_heads

        self.to_bias = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True))

        basic_module_init(self)

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        bias = self.to_bias(self.delta)
        bias = bias[self.index]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = bias.reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = bias.permute(1, 0).contiguous().reshape(self.num_heads, N1, N2)
        return bias


class WindowRelativeScoreBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        self.num_heads = num_heads
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2

        index, _ = _gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index.to(torch.int32))
        if num_heads is None:
            self.bias = nn.Parameter(torch.zeros((index.max() + 1,), dtype=torch.float32))
        else:
            self.bias = nn.Parameter(torch.zeros((num_heads, index.max() + 1), dtype=torch.float32))

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = self.bias[self.index].reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = self.bias[:, self.index].reshape(self.num_heads, N1, N2)

        return bias


class WindowDistanceScoreBias(nn.Module):
    def __init__(self, window_size, max_distance=None, num_heads=None):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))

        mask = None
        if num_heads is not None:
            distance = window_distance_matrix(self.window_size)
            distance = distance.expand(num_heads, *distance.shape)
            distance_bias = (1.0 + distance).log().neg()
            self.register_buffer("distance_bias", distance_bias)
            self.scale_bias = nn.Parameter(torch.zeros((num_heads, 1, 1), dtype=torch.float32))

            if max_distance is not None:
                if isinstance(max_distance, (list, tuple)):
                    if len(max_distance) != num_heads:
                        assert num_heads % len(max_distance) == 0
                        max_distance = max_distance * (num_heads // len(max_distance))
                    max_distance = torch.tensor(max_distance, dtype=torch.float32).view(num_heads, 1, 1)
                    mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))
                else:
                    mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))
        else:
            distance = window_distance_matrix(self.window_size)
            distance_bias = (1.0 + distance).log().neg()
            self.register_buffer("distance_bias", distance_bias)
            self.scale_bias = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

            if max_distance is not None:
                mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self):
        scale = self.scale_bias.exp()
        # print(self.window_size, scale.flatten().tolist())
        bias = self.distance_bias * scale
        if self.mask is not None:
            bias = bias + self.mask

        return bias


@torch.no_grad()
def window_distance_matrix(window_size):
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    else:
        window_size = window_size

    N = window_size[0] * window_size[1]
    positions = torch.stack(
        torch.meshgrid(torch.arange(0, window_size[0]),
                       torch.arange(0, window_size[1]), indexing="ij"), dim=2).reshape(N, 2)

    positions = positions.to(torch.float32)
    distance = torch.cat([((positions[i].view(1, 2) - positions) ** 2).sum(dim=1) ** 0.5
                          for i in range(positions.shape[0])], dim=0)
    distance = distance.view(N, N)
    return distance


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


def _test_spatial_reduction():
    import time
    kernel_size = 2
    for window_size in (8, 12, 16, 24, 32, 48):
        x = torch.zeros((16, 64, 96, 96)).cuda()
        mha1 = WindowMHA2d(64, 4, window_size=window_size).cuda().eval()
        mha2 = WindowSpatialReductionMHA2d(64, 4, window_size=window_size,
                                           kernel_size=kernel_size, reduction=2).cuda().eval()
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
    from . flex_attention import WindowNeighborhoodMHA2d

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


def _test_bias2():
    bias = WindowRelativeScoreBias(window_size=3, num_heads=4)
    bias()
    bias = WindowRelativeScoreBias(window_size=3)
    bias()


if __name__ == "__main__":
    # _test_spatial_reduction()
    _test_neighborhood()
    _test_bias()
    _test_bias2()
    pass
