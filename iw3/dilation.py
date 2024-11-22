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
    assert x.ndim == 4
    max_v = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    min_v = F.max_pool2d(x.neg(), kernel_size=3, stride=1, padding=1).neg()
    range_v = max_v - min_v
    range_c = range_v - range_v.mean(dim=[1, 2, 3], keepdim=True)
    range_s = range_c.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt()
    w = (range_c / (range_s + 1e-6)).clamp(-3, 3)
    w_min, w_max = w.amin(dim=[1, 2, 3], keepdim=True), w.amax(dim=[1, 2, 3], keepdim=True)
    w = (w - w_min) / ((w_max - w_min) + 1e-6)

    return w


@torch.inference_mode()
def dilate_edge(x, n):
    for _ in range(n):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2)
        x = (x * (1 - w)) + (x2 * w)

    return x


if __name__ == "__main__":
    import time
    import torchvision.io as IO
    import torchvision.transforms.functional as TF

    x1 = (IO.read_image("cc0/depth/dog.png") / 65535.0).mean(dim=0, keepdim=True)
    x2 = (IO.read_image("cc0/depth/light_house.png") / 65535.0).mean(dim=0, keepdim=True)
    x = torch.stack([x1, x2])
    z = edge_weight(x)
    TF.to_pil_image(z[0]).show()
    time.sleep(2)
    TF.to_pil_image(z[1]).show()
    time.sleep(2)

    z = dilate_edge(x, 1)
    TF.to_pil_image(z[0]).show()
    time.sleep(2)
    TF.to_pil_image(z[1]).show()
    time.sleep(2)
