import torch
import torch.nn.functional as F
from nunif.models import Model


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights


def normalize(x):
    return x * 2. - 1.


def modcrop(x, n):
    if x.shape[2] % n != 0:
        unpad = x.shape[2] % n // 2
        x = F.pad(x, (-unpad,) * 4)
    return x


def modpad(x, n):
    rem = n - input.shape[2] % n
    pad1 = rem // 2
    pad2 = rem - pad1
    x = F.pad(x, (pad1, pad2, pad1, pad2))
    return x


def fit_to_size(x, cond):
    dh = cond.shape[2] - x.shape[2]
    dw = cond.shape[3] - x.shape[3]
    assert dh >= 0 and dw >= 0
    pad_h, pad_w = dh // 2, dw // 2
    if pad_h > 0 or pad_w > 0:
        cond = F.pad(cond, (-pad_w, -pad_w, -pad_h, -pad_h))
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
