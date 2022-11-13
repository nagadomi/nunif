import torch


def _hflip(x):
    return torch.flip(x, (2,))


def _vflip(x):
    return torch.flip(x, (1,))


def _tr_f(x):
    return torch.rot90(x, 1, (1, 2))


def _itr_f(x):
    return torch.rot90(x, -1, (1, 2))


def tta_split(x):
    assert (isinstance(x, torch.Tensor) and x.dim() == 3)
    x_hflip = _hflip(x)
    x_vflip = _vflip(x)
    x_vflip_hflip = _hflip(x_vflip)
    x_tr = _tr_f(x)
    x_tr_hflip = _hflip(x_tr)
    x_tr_vflip = _vflip(x_tr)
    x_tr_vflip_hflip = _hflip(x_tr_vflip)

    return (x, x_hflip, x_vflip, x_vflip_hflip,
            x_tr, x_tr_hflip, x_tr_vflip, x_tr_vflip_hflip)


def tta_merge(xs):
    (x, x_hflip, x_vflip, x_vflip_hflip,
     x_tr, x_tr_hflip, x_tr_vflip, x_tr_vflip_hflip) = xs

    avg = x.clone()
    avg += _hflip(x_hflip)
    avg += _vflip(x_vflip)
    avg += _vflip(_hflip(x_vflip_hflip))
    avg += _itr_f(x_tr)
    avg += _itr_f(_hflip(x_tr_hflip))
    avg += _itr_f(_vflip(x_tr_vflip))
    avg += _itr_f(_vflip(_hflip(x_tr_vflip_hflip)))
    avg *= 1 / 8.0

    return torch.clamp_(avg, 0, 1)
