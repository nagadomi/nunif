from . tta import tta_split, tta_merge

_clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)


def quantize256(float_tensor):
    dest = (float_tensor + _clip_eps8).mul_(255.0)
    return dest.clamp_(0, 255).byte()

