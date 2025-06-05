import torch
import torch.nn.functional as F
import math
from nunif.device import device_is_mps


def equirectangular_projection(c, device="cpu"):
    # CHW
    c = c.to(device)
    h, w = c.shape[1:]
    max_edge = max(h, w)
    output_size = max_edge + max_edge // 2
    pad_w = (output_size - w) // 2
    pad_h = (output_size - h) // 2
    c = F.pad(c, (pad_w, pad_w, pad_h, pad_h),
              mode="constant")

    h, w = c.shape[1:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                          torch.linspace(-1, 1, w, device=device), indexing="ij")

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = (max_edge / output_size) * torch.tan(azimuth)
    mesh_y = (max_edge / output_size) * (torch.tan(elevation) / torch.cos(azimuth))
    grid = torch.stack((mesh_x, mesh_y), 2)

    if device_is_mps(c.device):
        # MPS does not support bicubic
        mode = "bilinear"
    else:
        mode = "bicubic"

    z = F.grid_sample(c.unsqueeze(0),
                      grid.unsqueeze(0),
                      mode=mode, padding_mode="zeros",
                      align_corners=True).squeeze(0)
    z = torch.clamp(z, 0, 1)

    return z
