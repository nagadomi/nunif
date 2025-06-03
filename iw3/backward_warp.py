import torch
from .mapper import get_mapper
from nunif.device import autocast, device_is_mps
import torch.nn.functional as F


def make_divergence_feature_value(divergence, convergence, image_width):
    # assert image_width <= 2048
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0

    return divergence_feature_value, convergence_feature_value


def make_input_tensor(c, depth, divergence, convergence,
                      image_width, mapper="pow2", preserve_screen_border=False):
    depth = depth.squeeze(0)  # CHW -> HW
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = torch.full_like(depth, divergence_value, device=depth.device)
    convergence_feat = torch.full_like(depth, convergence_value, device=depth.device)

    if preserve_screen_border:
        # Force set screen border parallax to zero.
        # Note that this does not work with tiled rendering (training code)
        border_pix = round(divergence * 0.75 * 0.01 * image_width * (depth.shape[-1] / image_width))
        if border_pix > 0:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, device=depth.device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, device=depth.device)
            divergence_feat[:, :border_pix] = (border_weight_l[None, :].expand_as(divergence_feat[:, :border_pix]) *
                                               divergence_feat[:, :border_pix])
            divergence_feat[:, -border_pix:] = (border_weight_r[None, :].expand_as(divergence_feat[:, -border_pix:]) *
                                                divergence_feat[:, -border_pix:])
            convergence_feat[:, :border_pix] = (border_weight_l[None, :].expand_as(convergence_feat[:, :border_pix]) *
                                                convergence_feat[:, :border_pix])
            convergence_feat[:, -border_pix:] = (border_weight_r[None, :].expand_as(convergence_feat[:, -border_pix:]) *
                                                 convergence_feat[:, -border_pix:])

    if c is not None:
        w, h = c.shape[2], c.shape[1]
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, h, device=c.device),
                                        torch.linspace(-1, 1, w, device=c.device), indexing="ij")
        grid = torch.stack((mesh_x, mesh_y), 2)
        grid = grid.permute(2, 0, 1)  # CHW
        return torch.cat([
            c,
            depth.unsqueeze(0),
            divergence_feat.unsqueeze(0),
            convergence_feat.unsqueeze(0),
            grid,
        ], dim=0)
    else:
        return torch.cat([
            depth.unsqueeze(0),
            divergence_feat.unsqueeze(0),
            convergence_feat.unsqueeze(0),
        ], dim=0)


def backward_warp(c, grid, delta, delta_scale):
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    if device_is_mps(c.device):
        # MPS does not support bicubic and border
        mode = "bilinear"
        padding_mode = "reflection"
    else:
        mode = "bicubic"
        padding_mode = "border"

    z = F.grid_sample(c, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    z = torch.clamp(z, 0, 1)
    return z


def make_grid(batch, width, height, device):
    # TODO: xpu: torch.meshgrid causes fallback from XPU to CPU, but it is faster to simply do nothing
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                    torch.linspace(-1, 1, width, device=device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).expand(batch, 1, height, width)
    mesh_x = mesh_x.reshape(1, 1, height, width).expand(batch, 1, height, width)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    return grid


def apply_divergence_grid_sample(c, depth, divergence, convergence, synthetic_view):
    assert synthetic_view in {"both", "right", "left"}
    # BCHW
    B, _, H, W = depth.shape

    if synthetic_view != "both":
        divergence = divergence * 2

    shift_size = divergence * 0.01
    index_shift = depth * shift_size - (shift_size * convergence)
    delta = torch.cat([index_shift, torch.zeros_like(index_shift)], dim=1)
    grid = make_grid(B, W, H, c.device)

    if synthetic_view == "both":
        left_eye = backward_warp(c, grid, -delta, 1)
        right_eye = backward_warp(c, grid, delta, 1)
    elif synthetic_view == "right":
        left_eye = c
        right_eye = backward_warp(c, grid, delta, 1)
    elif synthetic_view == "left":
        left_eye = backward_warp(c, grid, -delta, 1)
        right_eye = c

    return left_eye, right_eye


def apply_divergence_nn_LR(model, c, depth, divergence, convergence, steps,
                           mapper, synthetic_view, preserve_screen_border, enable_amp):
    assert synthetic_view in {"both", "right", "left"}
    steps = 1 if steps is None else steps

    if getattr(model, "symmetric", False):
        left_eye, right_eye = apply_divergence_nn_symmetric(
            model, c, depth, divergence, convergence,
            mapper=mapper, synthetic_view=synthetic_view, enable_amp=enable_amp)
    else:
        if synthetic_view == "both":
            left_eye = apply_divergence_nn(model, c, depth, divergence, convergence, steps,
                                           mapper=mapper, shift=-1,
                                           preserve_screen_border=preserve_screen_border,
                                           enable_amp=enable_amp)
            right_eye = apply_divergence_nn(model, c, depth, divergence, convergence, steps,
                                            mapper=mapper, shift=1,
                                            preserve_screen_border=preserve_screen_border,
                                            enable_amp=enable_amp)
        elif synthetic_view == "right":
            left_eye = c
            right_eye = apply_divergence_nn(model, c, depth, divergence * 2, convergence, steps,
                                            mapper=mapper, shift=1,
                                            preserve_screen_border=preserve_screen_border,
                                            enable_amp=enable_amp)
        elif synthetic_view == "left":
            left_eye = apply_divergence_nn(model, c, depth, divergence * 2, convergence, steps,
                                           mapper=mapper, shift=-1,
                                           preserve_screen_border=preserve_screen_border,
                                           enable_amp=enable_amp)
            right_eye = c

    return left_eye, right_eye


def apply_divergence_nn(model, c, depth, divergence, convergence, steps,
                        mapper, shift, preserve_screen_border, enable_amp):
    # BCHW
    assert model.delta_output
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    divergence_step = divergence / steps
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    depth_warp = depth
    delta_steps = []
    for j in range(steps):
        x = torch.stack([make_input_tensor(None, depth_warp[i],
                                           divergence=divergence_step,
                                           convergence=convergence,
                                           image_width=W,
                                           mapper=mapper,
                                           preserve_screen_border=preserve_screen_border)
                         for i in range(depth_warp.shape[0])])
        with autocast(device=depth.device, enabled=enable_amp):
            delta = model(x)
        delta_steps.append(delta)
        if j + 1 < steps:
            depth_warp = backward_warp(depth_warp, grid, delta, delta_scale)

    c_warp = c
    for delta in delta_steps:
        c_warp = backward_warp(c_warp, grid, delta, delta_scale)
    z = c_warp

    if shift > 0:
        z = torch.flip(z, (3,))

    return z


def apply_divergence_nn_symmetric(model, c, depth, divergence, convergence,
                                  mapper, synthetic_view, enable_amp):
    # BCHW
    assert synthetic_view in {"both", "right", "left"}
    assert model.delta_output
    assert model.symmetric
    B, _, H, W = depth.shape

    if synthetic_view != "both":
        divergence *= 2

    x = torch.stack([make_input_tensor(None, depth[i],
                                       divergence=divergence,
                                       convergence=convergence,
                                       image_width=W,
                                       mapper=mapper)
                     for i in range(depth.shape[0])])
    with autocast(device=depth.device, enabled=enable_amp):
        delta = model(x)
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)

    if synthetic_view == "both":
        left_eye = backward_warp(c, grid, delta, delta_scale)
        right_eye = backward_warp(c, grid, -delta, delta_scale)
    elif synthetic_view == "right":
        left_eye = c
        right_eye = backward_warp(c, grid, -delta, delta_scale)
    elif synthetic_view == "left":
        left_eye = backward_warp(c, grid, delta, delta_scale)
        right_eye = c

    return left_eye, right_eye
