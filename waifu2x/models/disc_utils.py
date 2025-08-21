import torch
import torch.nn.functional as F
from nunif.models import Model
from nunif.modules.pad import get_crop_size, get_pad_size, get_fit_pad_size
from nunif.modules.reflection_pad2d import reflection_pad2d_naive


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights


def normalize(x):
    return x * 2. - 1.


def to_y(rgb):
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    return r * 0.299 + g * 0.587 + b * 0.114


def modcrop(x, n):
    unpad = get_crop_size(x, n)
    x = F.pad(x, unpad)
    return x


def modpad(x, n):
    pad = get_pad_size(x, n)
    x = reflection_pad2d_naive(x, pad, detach=True)
    return x


def fit_to_size(x, cond):
    pad = get_fit_pad_size(cond, x)
    cond = reflection_pad2d_naive(cond, pad, detach=True)
    return cond


def fit_to_size_x(x, cond, scale_factor):
    dh = cond.shape[2] - x.shape[2] // scale_factor
    dw = cond.shape[3] - x.shape[3] // scale_factor
    assert dh >= 0 and dw >= 0
    pad_h, pad_w = dh // 2, dw // 2
    if pad_h > 0 or pad_w > 0:
        cond = F.pad(cond, (-pad_w, -pad_w, -pad_h, -pad_h))
    return cond


def add_bias(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x + cond


def apply_scale(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cond = torch.sigmoid(cond)
    return x * cond


def apply_scale_bias(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    bias, scale = cond.chunk(2, dim=1)
    scale = torch.sigmoid(scale)
    return x * scale + bias


def apply_patch_project(x, cond):
    if cond.shape[-2:] != x.shape[-2:]:
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (x * cond).sum(dim=1, keepdim=True) * (x.shape[1] ** 0.5)


def apply_project(x, cond):
    if cond.shape[-1] != 1:
        cond = F.adaptive_avgpool2d(cond, (1, 1))
    return (x * cond).sum(dim=1, keepdim=True) * (x.shape[1] ** 0.5)


def bench(name, compile=False):
    from nunif.models import create_model
    import time

    N = 20
    B = 4
    S = (256, 256)
    device = "cuda:0"

    model = create_model(name).to(device).eval()
    if compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    c = torch.zeros((B, 3, *S)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x, c, scale_factor=4)
        print(z.shape)
        param = sum([p.numel() for p in model.parameters()])
        print(model.name, f"{param:,}", f"compile={compile}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z, *_ = model(x, c, scale_factor=4)
    torch.cuda.synchronize()
    et = time.time() - t
    print(et, 1 / (et / (B * N)), "FPS")
