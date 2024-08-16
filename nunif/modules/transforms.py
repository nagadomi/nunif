import torch
import torch.nn.functional as F
import math
from functools import lru_cache


# differentiable transforms for loss function


def rotate_grid(batch, height, width, angle, device):
    with torch.no_grad():
        angle = math.radians(angle)
        py, px = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                torch.linspace(-1, 1, width, device=device), indexing="ij")
        mesh_x = px * math.cos(angle) - py * math.sin(angle)
        mesh_y = px * math.sin(angle) + py * math.cos(angle)
        grid = torch.stack((mesh_x, mesh_y), 2).unsqueeze(0).repeat(batch, 1, 1, 1).detach()
    return grid


@lru_cache
def rotate_grid_cache(batch, height, width, angle, device):
    return rotate_grid(batch, height, width, angle, device)


PAD_MODE_NN = {
    "zeros": "constant",
    "reflection": "reflect",
    "border": "replicate",
}


def diff_rotate(x, angle, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False, cache=True):
    # x: BCHW
    B, _, H, W = x.shape
    if expand:
        pad_h = (int(2 ** 0.5 * H) - H) // 2 + 1
        pad_w = (int(2 ** 0.5 * W) - W) // 2 + 1
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h),
                  mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0)
        B, _, H, W = x.shape

    if cache:
        grid = rotate_grid_cache(B, H, W, angle, x.device)
    else:
        grid = rotate_grid(B, H, W, angle, x.device)

    x = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return x


def diff_random_rotate(x, angle=45, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False):
    B, _, H, W = x.shape
    angle = (torch.rand((B,), device=x.device) * 2 - 1) * angle
    # FIXME: remove loop
    return torch.cat([diff_rotate(x[i:i + 1, :, :, :], angle=angle[i].item(),
                                  mode=mode, padding_mode=padding_mode,
                                  align_corners=align_corners, expand=expand, cache=False)
                      for i in range(B)], dim=0)


def diff_random_rotate_pair(x, y, angle=45, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False):
    B, _, H, W = x.shape
    angle = (torch.rand((B,), device=x.device) * 2 - 1) * angle
    xy = torch.stack((x, y), dim=1)
    xys = []
    for i in range(B):
        xys.append(diff_rotate(
            xy[i], angle=angle[i].item(),
            mode=mode, padding_mode=padding_mode,
            align_corners=align_corners, expand=expand, cache=False))
    x = torch.stack([xyi[0] for xyi in xys], dim=0)
    y = torch.stack([xyi[1] for xyi in xys], dim=0)

    return x, y


def diff_translate(x, x_shift, y_shift, padding_mode="zeros"):
    # NOTE: padded values with reflect or replicate have copied gradients.
    #       there may be cases where that is undesirable.
    return F.pad(x, (x_shift, -x_shift, y_shift, -y_shift),
                 mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0)


def diff_random_translate(x, ratio=0.15, padding_mode="zeros"):
    B, _, H, W = x.shape
    x_shift = torch.randint(low=int(-W * ratio), high=int(W * ratio), size=(B,), device=x.device)
    y_shift = torch.randint(low=int(-H * ratio), high=int(H * ratio), size=(B,), device=x.device)
    # FIXME: remove loop
    return torch.cat([diff_translate(x[i:i + 1, :, :, :], x_shift[i], y_shift[i],
                                     padding_mode=padding_mode) for i in range(B)], dim=0)


def _test_rotate():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)
    # TF.to_pil_image(x[0]).show()
    print(x.shape)

    z = diff_rotate(x, 45, expand=False)
    TF.to_pil_image(z[0]).show()
    time.sleep(0.5)

    z = diff_rotate(x, 45, expand=True, padding_mode="reflection")
    TF.to_pil_image(z[0]).show()

    x = x.repeat(4, 1, 1, 1)
    z = diff_random_rotate(x, padding_mode="reflection")
    for i in range(4):
        TF.to_pil_image(z[i]).show()
        time.sleep(1)


def _test_translate():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)

    x = x.repeat(4, 1, 1, 1)
    z = diff_random_translate(x, padding_mode="reflection")
    for i in range(4):
        TF.to_pil_image(z[i]).show()
        time.sleep(1)


def _bench_random_rotate_pair():
    import time
    N = 100
    B = 16
    x = torch.randn((B, 3, 512, 512)).cuda()
    y = torch.randn((B, 3, 512, 512)).cuda()

    t = time.time()
    for _ in range(N):
        diff_random_rotate_pair(x, y)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")


if __name__ == "__main__":
    # _test_rotate()
    # _test_translate()
    _bench_random_rotate_pair()
