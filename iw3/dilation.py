import torch.nn.functional as F
import torch


def edge_dilation_parse(edge_dilation):
    if isinstance(edge_dilation, list):
        if len(edge_dilation) == 0:
            x = y = 0
        elif len(edge_dilation) == 1:
            x = y = edge_dilation[0]
        else:
            x = edge_dilation[0]
            y = edge_dilation[1]
    elif isinstance(edge_dilation, int):
        x = y = edge_dilation
    else:
        x = y = 0

    return x, y


def edge_dilation_is_enabled(edge_dilation):
    x, y = edge_dilation_parse(edge_dilation)
    return x != 0 or y != 0


def gaussian_blur(x):
    kernel = torch.tensor([
        [21, 31, 21],
        [31, 48, 31],
        [21, 31, 21],
    ], dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3) / 256.0
    x = F.pad(x, [1] * 4, mode="replicate")
    x = F.conv2d(x, weight=kernel, bias=None, stride=1, padding=0, groups=1)
    return x


def dilate(mask, kernel_size=3):
    if isinstance(kernel_size, (list, tuple)):
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def erode(mask, kernel_size=3):
    if isinstance(kernel_size, (list, tuple)):
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


def closing(mask, kernel_size=3, n_iter=2):
    mask = mask.float()
    for _ in range(n_iter):
        mask = dilate(mask, kernel_size=kernel_size)
    for _ in range(n_iter):
        mask = erode(mask, kernel_size=kernel_size)

    return mask


def dilate_outer(mask, n_iter, base_width=None):
    # right view base
    if n_iter <= 0:
        return mask

    mask_dtype = mask.dtype
    mask = mask.bool()

    if base_width is not None:
        n_iter = max(round(mask.shape[-1] / base_width * n_iter), 1)

    for i in range(n_iter):
        mask = mask | F.pad(mask, (1, 0, 0, 0))[:, :, :, :-1]

    return mask.to(mask_dtype)


def dilate_inner(mask, n_iter, base_width=None):
    # right view base
    if n_iter <= 0:
        return mask

    mask_dtype = mask.dtype
    mask = mask.bool()

    if base_width is not None:
        n_iter = max(round(mask.shape[-1] / base_width * n_iter), 1)

    for i in range(n_iter):
        mask = mask | F.pad(mask, (0, 1, 0, 0))[:, :, :, 1:]

    return mask.to(mask_dtype)


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
    x_iter, y_iter = edge_dilation_parse(n)
    xy_iter = min(x_iter, y_iter)
    x_iter = x_iter - xy_iter
    y_iter = y_iter - xy_iter

    # print(xy_iter, x_iter, y_iter)

    for _ in range(xy_iter):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2, kernel_size=(3, 3))
        x = (x * (1 - w)) + (x2 * w)

    for _ in range(y_iter):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2, kernel_size=(3, 1))
        x = (x * (1 - w)) + (x2 * w)

    for _ in range(x_iter):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2, kernel_size=(1, 3))
        x = (x * (1 - w)) + (x2 * w)

    return x


def mask_closing(mask, kernel_size=3, n_iter=2):
    mask = mask_org = mask.float()
    mask = closing(mask, kernel_size=kernel_size, n_iter=n_iter)

    # Add erased isolated pixels
    mask = (mask + mask_org).clamp(0, 1)

    return mask


def _test_dialte_edge():
    import time
    import torchvision.io as io
    import torchvision.transforms.functional as TF

    x1 = (io.read_image("cc0/depth/dog.png") / 65535.0).mean(dim=0, keepdim=True)
    x2 = (io.read_image("cc0/depth/light_house.png") / 65535.0).mean(dim=0, keepdim=True)
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


def _test_mask():
    import time
    import torchvision.io as io
    import torchvision.transforms.functional as TF

    mask = io.read_image("cc0/mask/dog.png") / 255.0
    mask = mask.unsqueeze(0)
    inner = dilate_inner(mask, n_iter=2, base_width=mask.shape[-1] // 2)[0]
    outer = dilate_outer(mask, n_iter=2)[0]

    TF.to_pil_image(mask[0]).show()
    time.sleep(2)
    TF.to_pil_image(inner).show()
    time.sleep(2)
    TF.to_pil_image(outer).show()


if __name__ == "__main__":
    # _test_dialte_edge()
    _test_mask()
