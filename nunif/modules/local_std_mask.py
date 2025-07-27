import torch
import torch.nn.functional as F


def compute_local_std_mask(y, kernel_size=15, max_std=0.1, min_weight=0.1):
    B, C, H, W = y.shape
    padded = F.pad(y.detach(), ((kernel_size - 1) // 2,) * 4, mode="reflect")
    patch = F.unfold(padded, kernel_size=kernel_size, stride=1, padding=0)
    patch = patch.reshape(B, C, kernel_size * kernel_size, H * W)
    patch_std = torch.std(patch, dim=2, correction=0, keepdim=False)
    patch_std = torch.mean(patch_std, dim=1, keepdim=True)
    patch_weight = F.fold(patch_std, kernel_size=1, output_size=(H, W))
    patch_weight = (patch_weight.clamp(max=max_std) / max_std).clamp(min=min_weight)
    return patch_weight


def local_std_mask(x, y, kernel_size=15, max_std=0.1, min_weight=0.1):
    assert x.shape == y.shape
    weight = compute_local_std_mask(y, kernel_size=kernel_size, max_std=max_std, min_weight=min_weight)
    x = x * weight + x.detach() * (1 - weight)
    return x


def _test():
    import torchvision.io as io
    import torchvision.transforms.functional as TF
    x = io.read_image("cc0/320/light_house.png") / 255.0
    z = compute_local_std_mask(x.unsqueeze(0)).squeeze(0)
    TF.to_pil_image(z).show()


if __name__ == "__main__":
    _test()
