import math

import torch
import torch.nn.functional as F

from nunif.device import device_is_mps

# Cache invariant projection grids by resolution and device to avoid repeated
# trigonometric (tan/cos) and meshgrid allocations on every frame.
_GRID_CACHE: dict[tuple[int, int, torch.device, torch.dtype], tuple[int, int, torch.Tensor]] = {}


def get_projection_grid(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> tuple[int, int, torch.Tensor]:
    key = (h, w, device, dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    max_edge = max(h, w)
    output_size = max_edge + max_edge // 2
    pad_w = (output_size - w) // 2
    pad_h = (output_size - h) // 2

    padded_h = h + pad_h * 2
    padded_w = w + pad_w * 2

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, padded_h, device=device, dtype=dtype),
        torch.linspace(-1, 1, padded_w, device=device, dtype=dtype),
        indexing="ij",
    )

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = (max_edge / output_size) * torch.tan(azimuth)
    mesh_y = (max_edge / output_size) * (torch.tan(elevation) / torch.cos(azimuth))
    grid = torch.stack((mesh_x, mesh_y), 2).unsqueeze(0)

    if len(_GRID_CACHE) >= 8:
        _GRID_CACHE.clear()

    _GRID_CACHE[key] = (pad_w, pad_h, grid)
    return pad_w, pad_h, grid


def equirectangular_projection(c: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
    if device is not None:
        c = c.to(device)

    h, w = c.shape[-2:]
    pad_w, pad_h, grid = get_projection_grid(h, w, c.device, c.dtype)
    c = F.pad(c, (pad_w, pad_w, pad_h, pad_h), mode="constant")

    if device_is_mps(c.device):
        # MPS does not support bicubic
        mode = "bilinear"
    else:
        mode = "bicubic"

    batch = c.ndim == 4
    if not batch:
        c = c.unsqueeze(0)

    grid_input = grid if c.shape[0] == 1 else grid.expand(c.shape[0], -1, -1, -1)
    z = F.grid_sample(c, grid_input, mode=mode, padding_mode="zeros", align_corners=True)
    z = torch.clamp(z, 0, 1)

    if not batch:
        z = z.squeeze(0)

    return z
