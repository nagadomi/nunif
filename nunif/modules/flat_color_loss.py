import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def get_flat_color_mask(y, size=8, eps=1e-5):
    # mul_mask: 1.0: true, 0.0: false
    def pad_size(value, mod):
        return 0 if value % mod == 0 else (mod - value % mod)

    height, width = y.shape[2:]
    pad_h1 = pad_size(height, size) // 2
    pad_h2 = pad_size(height, size) - pad_h1
    pad_w1 = pad_size(width, size) // 2
    pad_w2 = pad_size(width, size) - pad_w1

    y = F.pad(y, (pad_w1, pad_w2, pad_h1, pad_h2), mode="constant")
    cell = F.interpolate(y, scale_factor=1.0 / size, mode="nearest")
    cell = F.interpolate(cell, scale_factor=size, mode="nearest")

    diff = (y - cell).abs()
    diff_max = F.max_pool2d(diff, kernel_size=size, stride=size).amax(dim=1, keepdim=True)
    mask = (diff_max < eps).to(y.dtype)
    mask = F.interpolate(mask, scale_factor=size, mode="nearest")
    mask = F.pad(mask, (-pad_w1, -pad_w2, -pad_h1, -pad_h2), mode="constant")

    return mask.detach()


def flat_color_loss(input, target, size=8):
    mask = get_flat_color_mask(target, size=size)
    l2_loss = F.mse_loss(input, target, reduction="none") * mask
    # l2_loss = (l2_loss.sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3])).mean()
    l2_loss = l2_loss.mean()
    return l2_loss


def flat_color_weighted_loss(input, target, size=8):
    l1_loss = F.l1_loss(input, target)
    l2_loss = flat_color_loss(input, target, size=size)

    return l1_loss + l2_loss


class FlatColorWeightedLoss(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.size = size

    def forward(self, input, target):
        return flat_color_weighted_loss(input, target, size=self.size)


def _test_mask():
    import argparse
    import torchvision.transforms.functional as TF
    import torchvision.io as io

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input file")
    parser.add_argument("--size", type=int, default=8, help="flat cell size")
    args = parser.parse_args()
    x = io.read_image(args.input, io.ImageReadMode.RGB)
    assert x.shape[0] == 3
    x = (x / 255.0).unsqueeze(0)

    mask = get_flat_color_mask(x, size=args.size)
    x = x * mask
    TF.to_pil_image(x[0]).show()


if __name__ == "__main__":
    _test_mask()
