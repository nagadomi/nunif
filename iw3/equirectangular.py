import math

import torch
import torch.nn.functional as F

from nunif.device import device_is_mps

# Cache invariant projection grids and active bounding boxes by resolution and device
# to avoid repeated trigonometric (tan/cos) and meshgrid allocations on every frame.
_GRID_CACHE: dict[
    tuple[int, int, int, int, torch.device, torch.dtype],
    tuple[torch.Tensor, int, int, int, int, int, int, int],
] = {}


def resolve_equirectangular_size(h: int, w: int) -> tuple[int, int]:
    max_edge = max(h, w)
    default_size = max_edge + max_edge // 2
    pad_w = (default_size - w) // 2
    pad_h = (default_size - h) // 2
    return h + pad_h * 2, w + pad_w * 2


def get_projection_grid_and_bounds(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    output_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, int, int, int, int, int, int, int]:
    padded_h, padded_w = resolve_equirectangular_size(h, w)
    out_h, out_w = output_size if output_size is not None else (padded_h, padded_w)
    key = (h, w, out_h, out_w, device, dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    max_edge = max(h, w)
    default_size = max_edge + max_edge // 2
    scale_x = (padded_w - 1) / (w - 1) if w > 1 else 1.0
    scale_y = (padded_h - 1) / (h - 1) if h > 1 else 1.0

    # Compute the active bounding box where |mesh_x| <= 1 and |mesh_y| <= 1
    # (plus a 4-pixel safety margin for the 4x4 bicubic interpolation kernel).
    # Outside this central region, padding_mode="zeros" is identically 0.0.
    coef_x = (max_edge / default_size) * scale_x
    coef_y = (max_edge / default_size) * scale_y
    margin_x = 8.0 / max(w - 1, 1)
    margin_y = 8.0 / max(h - 1, 1)
    max_norm_x = (2.0 / math.pi) * math.atan((1.0 + margin_x) / max(coef_x, 1e-6))
    max_norm_y = (2.0 / math.pi) * math.atan((1.0 + margin_y) / max(coef_y, 1e-6))

    x0 = max(0, int(math.floor((1.0 - max_norm_x) * 0.5 * (out_w - 1))) - 2)
    x1 = min(out_w, int(math.ceil((1.0 + max_norm_x) * 0.5 * (out_w - 1))) + 3)
    y0 = max(0, int(math.floor((1.0 - max_norm_y) * 0.5 * (out_h - 1))) - 2)
    y1 = min(out_h, int(math.ceil((1.0 + max_norm_y) * 0.5 * (out_h - 1))) + 3)
    if (x1 - x0) % 2 != 0:
        if x1 < out_w:
            x1 += 1
        elif x0 > 0:
            x0 -= 1
    if (y1 - y0) % 2 != 0:
        if y1 < out_h:
            y1 += 1
        elif y0 > 0:
            y0 -= 1

    # When downscaling to output_size < (padded_h, padded_w), project the active
    # bounding box at 2x resolution and downsample with antialias=True, matching
    # the original master branch quality at 6x lower pixel count.
    ss = 2 if (out_h < padded_h or out_w < padded_w) else 1
    if ss == 2:
        x_full = torch.linspace(-1, 1, out_w * 2, device=device, dtype=dtype)
        y_full = torch.linspace(-1, 1, out_h * 2, device=device, dtype=dtype)
        y, x = torch.meshgrid(y_full[y0 * 2 : y1 * 2], x_full[x0 * 2 : x1 * 2], indexing="ij")
    else:
        x_full = torch.linspace(-1, 1, out_w, device=device, dtype=dtype)
        y_full = torch.linspace(-1, 1, out_h, device=device, dtype=dtype)
        y, x = torch.meshgrid(y_full[y0:y1], x_full[x0:x1], indexing="ij")

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = coef_x * torch.tan(azimuth)
    mesh_y = coef_y * (torch.tan(elevation) / torch.cos(azimuth))
    grid = torch.stack((mesh_x, mesh_y), 2).unsqueeze(0)

    if len(_GRID_CACHE) >= 8:
        _GRID_CACHE.clear()

    result = (grid, y0, y1, x0, x1, ss, out_h, out_w)
    _GRID_CACHE[key] = result
    return result


def get_projection_grid(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    padded_h, padded_w = resolve_equirectangular_size(h, w)
    out_h, out_w = output_size if output_size is not None else (padded_h, padded_w)
    max_edge = max(h, w)
    default_size = max_edge + max_edge // 2
    scale_x = (padded_w - 1) / (w - 1) if w > 1 else 1.0
    scale_y = (padded_h - 1) / (h - 1) if h > 1 else 1.0

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, out_h, device=device, dtype=dtype),
        torch.linspace(-1, 1, out_w, device=device, dtype=dtype),
        indexing="ij",
    )

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = ((max_edge / default_size) * scale_x) * torch.tan(azimuth)
    mesh_y = ((max_edge / default_size) * scale_y) * (torch.tan(elevation) / torch.cos(azimuth))
    return torch.stack((mesh_x, mesh_y), 2).unsqueeze(0)


def equirectangular_projection(
    c: torch.Tensor,
    device: str | torch.device | None = None,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if device is not None:
        c = c.to(device)

    h, w = c.shape[-2:]
    grid, y0, y1, x0, x1, ss, out_h, out_w = get_projection_grid_and_bounds(
        h, w, c.device, c.dtype, output_size=output_size
    )

    if device_is_mps(c.device):
        # MPS does not support bicubic
        mode = "bilinear"
    else:
        mode = "bicubic"

    batch = c.ndim == 4
    if not batch:
        c = c.unsqueeze(0)

    grid_input = grid if c.shape[0] == 1 else grid.expand(c.shape[0], -1, -1, -1)
    crop = F.grid_sample(c, grid_input, mode=mode, padding_mode="zeros", align_corners=True)
    if ss > 1:
        crop = F.interpolate(
            crop,
            size=(y1 - y0, x1 - x0),
            mode=mode,
            align_corners=False,
            antialias=(mode == "bicubic"),
        )
    crop = torch.clamp(crop, 0, 1)

    if y0 == 0 and y1 == out_h and x0 == 0 and x1 == out_w:
        z = crop
    else:
        z = torch.zeros((c.shape[0], c.shape[1], out_h, out_w), dtype=c.dtype, device=c.device)
        z[:, :, y0:y1, x0:x1] = crop

    if not batch:
        z = z.squeeze(0)

    return z
