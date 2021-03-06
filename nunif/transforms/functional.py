import torch
from . tta import tta_split, tta_merge
from . import image_magick


_clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)


def quantize256(ft):
    return (ft + _clip_eps8).mul_(255.0).clamp_(0, 255).byte()


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


def rgb2y_matlab(rgb):
    """
    rgb2y for compatibility with SISR benchmarks
    y: 16-235
    """
    assert(isinstance(rgb, (torch.FloatTensor, torch.cuda.FloatTensor)))
    y = rgb.new_full((1, rgb.shape[1], rgb.shape[2]), fill_value=0)
    (y.add_(rgb[0], alpha=65.481).
     add_(rgb[1], alpha=128.553).
     add_(rgb[2], alpha=24.966).
     add_(16.0).clamp_(0, 255))
    return y.byte().float()


def negate(x):
    if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)):
        # uint8
        return (-x).add_(255)
    else:
        # float
        return (-x).add_(1.0)


def crop(x, i, j, h, w):
    return x[:, i:(i + h), i:(i + w)].clone()


def crop_mod(x, mod):
    h, w = x.shape[1], x.shape[2]
    h_mod = h % mod
    w_mod = w % mod
    return crop(x, 0, 0, h - h_mod, w - w_mod)
