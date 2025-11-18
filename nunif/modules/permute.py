import torch
import torch.nn as nn

"""
permute() utilities

NOTE:
 Not sure if contiguous() is needed,
 but it would be more efficient to do it when needed.
"""


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()


def bchw_to_bhwc(x):
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x):
    return x.permute(0, 3, 1, 2)


class BCHWToBHWC(nn.Module):
    def forward(self, x):
        return bchw_to_bhwc(x)


class BHWCToBCHW(nn.Module):
    def forward(self, x):
        return bhwc_to_bchw(x)


def pixel_unshuffle(x, window_size):
    """ reference implementation of F.pixel_unshuffle + non-square window
    """
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    SS = SH * SW
    assert H % SH == 0 and W % SW == 0

    oc = C * SS
    oh = H // SH
    ow = W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    # B, C, SH, SW, oh, ow
    x = x.permute(0, 1, 3, 5, 2, 4)
    # B, (C, SH, SW), oh, ow
    x = x.reshape(B, oc, oh, ow)

    return x


def pixel_shuffle(x, window_size):
    """ reference implementation of F.pixel_shuffle + non-square window
    """
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    SS = SH * SW
    assert C % SS == 0 and C % SS == 0

    oc = C // SS
    oh = H * SH
    ow = W * SW
    x = x.reshape(B, oc, SH, SW, H, W)
    # B, oc, H, SH, W, SW
    x = x.permute(0, 1, 4, 2, 5, 3)
    # B, oc, (H, SH), (W, SW)
    x = x.reshape(B, oc, oh, ow)

    return x


def bchw_to_bnc(x, window_size):
    # For sequential model, e.g. transformer, lstm
    # b = B * (h // window_size) * (w // window_size)
    # n = window_size * window_size
    # c = c
    # aka. window_partition
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert H % SH == 0 and W % SW == 0

    oh = H // SH
    ow = W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    # B, oh, ow, SH, SW, C
    x = x.permute(0, 2, 4, 3, 5, 1)
    # (B, SH, SW), (oh, ow), C
    x = x.reshape(B * oh * ow, SH * SW, C)

    return x


def bhwc_to_bnc(x, window_size):
    # NOTE: not optimized
    x = bhwc_to_bchw(x)
    x = bchw_to_bnc(x, window_size=window_size)
    return x


def bnc_to_bchw(x, out_shape, window_size):
    # reverse bchw_to_bnc
    B, N, C = x.shape
    OB, OC, OH, OW = out_shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert OH % SH == 0 and OW % SW == 0
    H = OH // SH
    W = OW // SW

    x = x.reshape(OB, H, W, SH, SW, C)
    # OB, C, H, SH, W, SW
    x = x.permute(0, 5, 1, 3, 2, 4)
    # OB, (H * SH), (W * SW), C
    x = x.reshape(OB, C, OH, OW)

    return x


def window_partition2d(x, window_size):
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert H % SH == 0 and W % SW == 0

    oh = H // SH
    ow = W // SW
    # B, C, oh, SH, ow, SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    # B, oh, ow, C, SH, SW
    x = x.permute(0, 2, 4, 1, 3, 5)
    # B, (oh, ow), C, SH, SW
    x = x.reshape(B, oh * ow, C, SH, SW)

    return x


def window_reverse2d(x, out_shape, window_size):
    # reverse window_reverse2d
    # x: B, N, C, H, W
    OB, OC, OH, OW = out_shape
    assert OC == x.shape[2]
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert OH % SH == 0 and OW % SW == 0
    H = OH // SH
    W = OW // SW
    x = x.reshape(OB, H, W, OC, SH, SW)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(OB, OC, OH, OW)
    return x


def bcdhw_to_bnc(x, window_size):
    """
    BCDHW -> BNC
    For sequential models (transformer, lstm) on 3D input
    x: (B, C, D, H, W)
    window_size: (SD, SH, SW)
    Returns: (B * (D//SD) * (H//SH) * (W//SW), SD*SH*SW, C)
    """
    B, C, D, H, W = x.shape
    SD, SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size, window_size]
    assert D % SD == 0 and H % SH == 0 and W % SW == 0

    OD = D // SD
    OH = H // SH
    OW = W // SW

    # (B, C, OD, SD, OH, SH, OW, SW)
    x = x.reshape(B, C, OD, SD, OH, SH, OW, SW)
    # (B, OD, OH, OW, SD, SH, SW, C)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    # (B*OD*OH*OW, SD*SH*SW, C)
    x = x.reshape(B * OD * OH * OW, SD * SH * SW, C)
    return x


def bnc_to_bcdhw(x, out_shape, window_size):
    """
    BNC -> BCDHW
    Reverse of bcdhw_to_bnc
    x: (B*OD*OH*OW, SD*SH*SW, C)
    out_shape: (B, C, D, H, W)
    window_size: (SD, SH, SW)
    Returns: (B, C, D, H, W)
    """
    OB, OC, OD_tot, OH_tot, OW_tot = out_shape
    SD, SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size, window_size]
    assert OD_tot % SD == 0 and OH_tot % SH == 0 and OW_tot % SW == 0

    OD = OD_tot // SD
    OH = OH_tot // SH
    OW = OW_tot // SW

    BNC_B, N, C = x.shape
    assert N == SD * SH * SW
    assert OB * OD * OH * OW == BNC_B
    assert OC == C

    # (B, OD, OH, OW, SD, SH, SW, C)
    x = x.reshape(OB, OD, OH, OW, SD, SH, SW, C)
    # (B, C, OD, SD, OH, SH, OW, SW)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    # back (B, C, D, H, W)
    x = x.reshape(OB, OC, OD_tot, OH_tot, OW_tot)
    return x


def kernel2d_to_conv2d_weight(in_channels, kernel):
    assert kernel.ndim == 2
    kernel = kernel.reshape(1, 1, *kernel.shape)
    kernel = kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()
    return kernel


def kernel1d_to_conv1d_weight(in_channels, kernel):
    assert kernel.ndim == 1
    kernel = kernel.reshape(1, 1, *kernel.shape)
    kernel = kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()
    return kernel


def _test_bhwc():
    src = x = torch.rand((4, 3, 2, 2))
    x = bchw_to_bhwc(x)
    assert x.shape == (4, 2, 2, 3)
    x = bhwc_to_bchw(x)
    assert x.shape == (4, 3, 2, 2)
    assert (x - src).abs().sum() == 0.
    print("pass _test_bhwc")


def _test_pixel_shuffle():
    import torch.nn.functional as F

    src = x = torch.rand((4, 3, 6, 6))
    x = pixel_unshuffle(x, 2)
    assert x.shape == (4, 3 * 2 * 2, 3, 3)
    x = pixel_shuffle(x, 2)
    assert x.shape == (4, 3, 6, 6)
    assert (x - src).abs().sum() == 0.

    x = pixel_unshuffle(x, (2, 3))
    assert x.shape == (4, 3 * 2 * 3, 3, 2)
    x = pixel_shuffle(x, (2, 3))
    assert x.shape == (4, 3, 6, 6)
    assert (x - src).abs().sum() == 0.

    # compatible
    x1 = pixel_unshuffle(x, 2)
    x2 = F.pixel_unshuffle(x, 2)
    assert x1.shape == x2.shape and (x1 - x2).abs().sum() == 0.

    x1 = pixel_shuffle(x1, 2)
    x2 = F.pixel_shuffle(x2, 2)
    assert x1.shape == x2.shape and (x1 - x2).abs().sum() == 0.
    print("pass _test_pixel_shuffle")


def _test_bnc():
    src = x = torch.rand((4, 3, 6, 6))
    original_shape = x.shape
    x = bchw_to_bnc(x, 2)
    assert x.shape == (4 * 3 * 3, 2 * 2, 3)
    x = bnc_to_bchw(x, original_shape, 2)
    assert src.shape == x.shape and (src - x).abs().sum() == 0

    x = bchw_to_bnc(x, (2, 3))
    assert x.shape == (4 * 3 * 2, 2 * 3, 3)
    x = bnc_to_bchw(x, original_shape, (2, 3))
    assert src.shape == x.shape and (src - x).abs().sum() == 0


def _test_3d_bnc():
    src = x = torch.rand((4, 3, 6, 6, 6))
    original_shape = x.shape
    x = bcdhw_to_bnc(x, 2)
    assert x.shape == (4 * 3 * 3 * 3, 2 * 2 * 2, 3)
    x = bnc_to_bcdhw(x, original_shape, 2)
    assert src.shape == x.shape and (src - x).abs().sum() == 0

    x = bcdhw_to_bnc(x, (2, 2, 3))
    assert x.shape == (4 * 3 * 3 * 2, 2 * 2 * 3, 3)
    x = bnc_to_bcdhw(x, original_shape, (2, 2, 3))
    assert src.shape == x.shape and (src - x).abs().sum() == 0



def _test_window():
    x = torch.rand((4, 3, 6, 6))
    y = window_partition2d(x, window_size=2)
    z = window_reverse2d(y, x.shape, window_size=2)
    assert x.shape == z.shape
    assert (x - z).abs().sum() == 0


if __name__ == "__main__":
    _test_bhwc()
    _test_pixel_shuffle()
    _test_bnc()
    _test_3d_bnc()
    _test_window()
