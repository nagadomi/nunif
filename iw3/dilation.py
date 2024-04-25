import torch.nn.functional as F
import torch


def gaussian_blur(x):
    kernel = torch.tensor([
        [21, 31, 21],
        [31, 48, 31],
        [21, 31, 21],
    ], dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3) / 256.0
    x = F.pad(x, [1] * 4, mode="replicate")
    x = F.conv2d(x, weight=kernel, bias=None, stride=1, padding=0, groups=1)
    return x


def dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def edge_weight(x):
    max_v = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    min_v = F.max_pool2d(x.neg(), kernel_size=3, stride=1, padding=1).neg()
    range_v = max_v.sub_(min_v)
    range_c = range_v.sub_(range_v.mean())
    range_s = range_c.pow(2).mean().add_(1e-6)
    w = torch.clamp(range_c.div_(range_s), -2, 2).add_(2).div_(4)
    w_min, w_max = w.min(), w.max()
    if w_max - w_min > 0:
        w = (w - w_min) / (w_max - w_min)
    else:
        w.fill_(0)

    return w


@torch.inference_mode()
def dilate_edge(x, n):
    for _ in range(n):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2)
        x = (x * (1 - w)) + (x2 * w)

    return x
