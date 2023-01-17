import torch
from torch.nn import functional as F


_clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)
_clip_eps16 = (1.0 / 65535.0) * 0.5 - (1.0e-7 * (1.0 / 65535.0) * 0.5)


def quantize256(ft):
    return (ft + _clip_eps8).mul_(255.0).clamp_(0, 255).to(torch.uint8)


def quantize65535(ft):
    return (ft + _clip_eps16).mul_(65535.0).clamp_(0, 65535).to(torch.int16)


def quantize256_f(ft):
    return (ft + _clip_eps8).mul_(255.0).clamp_(0, 255)


def _rgb2y(rgb):
    y = rgb.new_full((1, rgb.shape[1], rgb.shape[2]), fill_value=0)
    return (y.add_(rgb[0], alpha=0.299).
            add_(rgb[1], alpha=0.587).
            add_(rgb[2], alpha=0.114).clamp_(0, 1))


def rgb2y(rgb):
    if isinstance(rgb, (torch.ByteTensor, torch.cuda.ByteTensor)):
        return _rgb2y(rgb.float()).mul_(255).clamp_(0, 255).byte()
    else:
        return _rgb2y(rgb)


def to_grayscale(x):
    if x.shape[0] == 1:
        return x
    elif x.shape[0] == 3:
        return rgb2y(x)
    else:
        ValueError("Unknown channel format f{x.shape[0]}")


def rgb2y_matlab(rgb):
    """
    rgb2y for compatibility with SISR benchmarks
    y: 16-235
    """
    assert (isinstance(rgb, (torch.FloatTensor, torch.cuda.FloatTensor)))
    y = rgb.new_full((1, rgb.shape[1], rgb.shape[2]), fill_value=0)
    (y.add_(rgb[0], alpha=65.481).
     add_(rgb[1], alpha=128.553).
     add_(rgb[2], alpha=24.966).
     add_(16.0).add_(0.5 - 1e-6).clamp_(16, 235))
    return y.byte().float()


def negate(x):
    if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)):
        # uint8
        return (-x).add_(255)
    else:
        # float
        return (-x).add_(1.0)


def pad(x, pad, mode='constant', value=0):
    x = x.unsqueeze(0)
    return F.pad(x, pad, mode, value).squeeze(0)


def crop(x, i, j, h, w):
    return x[:, i:(i + h), j:(j + w)].clone()


def crop_ref(x, i, j, h, w):
    return x[:, i:(i + h), j:(j + w)]


def crop_mod(x, mod):
    h, w = x.shape[1], x.shape[2]
    h_mod = h % mod
    w_mod = w % mod
    return crop(x, 0, 0, h - h_mod, w - w_mod)
