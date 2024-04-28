import torch
import torch.nn.functional as F


def rgb_noise_like(base, level=2):
    assert level in {1, 2}
    noise = torch.randn_like(base)
    if level == 2:
        noise2 = torch.randn(base.shape[:-2] + (base.shape[-2] // 2, base.shape[-1] // 2),
                             dtype=base.dtype, device=base.device)
        if base.ndim == 3:
            noise2 = F.interpolate(noise2.unsqueeze(0), size=(base.shape[-2], base.shape[-1]), mode="nearest").squeeze(0)
        elif base.ndim == 4:
            noise2 = F.interpolate(noise2, size=(base.shape[-2], base.shape[-1]), mode="nearest")
        noise.mul_(0.5).add_(noise2, alpha=0.5)

    return noise


def apply_rgb_noise(rgb, noise, strength=0.2,
                    gamma=2.2,
                    light_decay=True, light_decay_strength=0.8):
    assert 0 <= light_decay_strength and light_decay_strength <= 1

    output = rgb ** gamma
    correlated_noise = noise * output
    if light_decay:
        light_decay = (1.0 - output).mul_(light_decay_strength).add_(1.0 - light_decay_strength)
        light_decay = light_decay.pow_(gamma)
    else:
        light_decay = torch.tensor(1.0, device=rgb.device)

    weight = light_decay.mul_(strength)
    output = output.add_(correlated_noise.mul_(weight))
    output = output.clamp_(0, 1).pow_(1.0 / gamma)
    return output
