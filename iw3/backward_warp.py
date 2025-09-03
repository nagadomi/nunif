import torch
from .mapper import get_mapper
from nunif.device import autocast, device_is_mps
import torch.nn.functional as F
from nunif.modules.replication_pad2d import replication_pad2d


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
        mode = "bilinear"
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


def apply_divergence_nn_LR(
        model, c, depth, divergence, convergence, steps,
        mapper,
        synthetic_view="both",
        preserve_screen_border=False,
        enable_amp=True,
        edge_dilation=2
):
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
                                           enable_amp=enable_amp, edge_dilation=edge_dilation)
            right_eye = apply_divergence_nn(model, c, depth, divergence, convergence, steps,
                                            mapper=mapper, shift=1,
                                            preserve_screen_border=preserve_screen_border,
                                            enable_amp=enable_amp, edge_dilation=edge_dilation)
        elif synthetic_view == "right":
            left_eye = c
            right_eye = apply_divergence_nn(model, c, depth, divergence * 2, convergence, steps,
                                            mapper=mapper, shift=1,
                                            preserve_screen_border=preserve_screen_border,
                                            enable_amp=enable_amp, edge_dilation=edge_dilation)
        elif synthetic_view == "left":
            left_eye = apply_divergence_nn(model, c, depth, divergence * 2, convergence, steps,
                                           mapper=mapper, shift=-1,
                                           preserve_screen_border=preserve_screen_border,
                                           enable_amp=enable_amp, edge_dilation=edge_dilation)
            right_eye = c

    return left_eye, right_eye


def apply_divergence_nn(
        model, c, depth, divergence, convergence, steps,
        mapper, shift,
        preserve_screen_border=False,
        enable_amp=True,
        edge_dilation=2
):
    if model.name == "sbs.mlbw":
        return apply_divergence_nn_delta_weight(
            model, c, depth,
            divergence=divergence,
            convergence=convergence,
            steps=steps,
            mapper=mapper,
            shift=shift,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
            edge_dilation=edge_dilation)
    else:
        return apply_divergence_nn_delta(
            model, c, depth,
            divergence=divergence,
            convergence=convergence,
            steps=steps,
            mapper=mapper,
            shift=shift,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp)


def pad_convergence_shift_05(c, divergence, convergence):
    if abs(float(convergence) - 0.5) > 1e-4:
        shift_size = divergence * 0.01 * 0.5 * c.shape[-1]
        convergence = convergence - 0.5
        convergence_shift = round(shift_size * convergence)
        c = replication_pad2d(c, (-convergence_shift, convergence_shift, 0, 0))
    return c


def apply_divergence_nn_delta(
        model, c, depth, divergence, convergence, steps,
        mapper, shift,
        preserve_screen_border=False,
        enable_amp=True
):
    # BCHW
    assert model.delta_output
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    divergence_step = divergence / steps
    grid = make_grid(B, W, H, c.device)
    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)
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


def pad_delta_y(delta_x):
    # interleave
    B, C, H, W = delta_x.shape
    delta = torch.stack([delta_x, torch.zeros_like(delta_x)], dim=2).reshape(B, C * 2, H, W)
    return delta


MLBW_DEBUG_OUTPUT = False
_mlbw_debug_count = 0


def mlbw_debug_output(z):
    import os
    import torchvision.transforms.functional as TF
    global _mlbw_debug_count
    _mlbw_debug_count += 1
    os.makedirs("tmp/mlbw_debug", exist_ok=True)
    z = torch.cat(z, dim=2)
    # only batch=0
    z = z.clamp(0, 1)[0]
    z = TF.to_pil_image(z).save(f"tmp/mlbw_debug/{_mlbw_debug_count}.png")


def apply_divergence_nn_delta_weight(
        model, c, depth, divergence, convergence, steps,
        mapper, shift,
        preserve_screen_border=False,
        enable_amp=True,
        return_mask=False,
        edge_dilation=2
):
    # BCHW
    assert model.delta_output
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    if True:  # if preserve_screen_border:
        input_convergence = convergence
        use_pad_convergence = False
    else:
        # use constant convergence
        # NOTE: In theory, this should be better, but because there is no noticeable difference
        #       and it could lead to confusion, it is currently not used.
        input_convergence = 0.5
        use_pad_convergence = True

    B, _, H, W = depth.shape
    x = torch.stack([make_input_tensor(None, depth[i],
                                       divergence=divergence,
                                       convergence=input_convergence,
                                       image_width=W,
                                       mapper=mapper,
                                       preserve_screen_border=preserve_screen_border)
                     for i in range(depth.shape[0])])
    with autocast(device=depth.device, enabled=enable_amp):
        if model.hole_mask:
            delta, layer_weight, hole_mask_logits = model(x)
        else:
            delta, layer_weight = model(x)
            hole_mask_logits = None

    if c.shape[2] != layer_weight.shape[2] or c.shape[3] != layer_weight.shape[3]:
        layer_weight = F.interpolate(layer_weight, size=c.shape[-2:],
                                     mode="bilinear", align_corners=True, antialias=True)

    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)
    delta = pad_delta_y(delta)
    grid = make_grid(B, W, H, c.device)
    z = torch.zeros_like(c)
    debug = []
    for i in range(model.num_layers):
        d = delta[:, i * 2:i * 2 + 2, :, :]
        w = layer_weight[:, i:i + 1, :, :]
        bw = backward_warp(c, grid, d, delta_scale) * w
        z += bw
        if MLBW_DEBUG_OUTPUT:
            debug.append(bw)

    if MLBW_DEBUG_OUTPUT and hole_mask_logits is not None:
        hole_mask_logits = F.interpolate(hole_mask_logits, size=c.shape[-2:],
                                         mode="bilinear", align_corners=True, antialias=False)
        hole_mask = torch.sigmoid(hole_mask_logits)
        debug.append(hole_mask.expand_as(c))

    if MLBW_DEBUG_OUTPUT:
        mlbw_debug_output(debug)
    del debug

    if use_pad_convergence:
        z = pad_convergence_shift_05(z, divergence, convergence)
        if hole_mask_logits is not None:
            hole_mask_logits = pad_convergence_shift_05(hole_mask_logits, divergence, convergence)
    z = z.clamp(0, 1)

    if shift > 0:
        z = z.flip((3,))
        if hole_mask_logits is not None:
            hole_mask_logits = hole_mask_logits.flip((3,))

    if return_mask:
        return z, hole_mask_logits
    else:
        if hole_mask_logits is not None:
            # hole fill for visualize
            hole_mask = postprocess_hole_mask(
                hole_mask_logits,
                target_size=c.shape[-2:],
                threshold=0.15,
                dilation=edge_dilation + 1,
            )
            z = z * (1 - hole_mask.float())

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


def postprocess_hole_mask(mask_logits, target_size, threshold, dilation):
    for _ in range(dilation):
        mask_logits = F.max_pool2d(mask_logits, kernel_size=(1, 3), stride=1, padding=(0, 1))

    if target_size != mask_logits.shape[-2:]:
        mask_logits = F.interpolate(mask_logits, size=target_size,
                                    mode="bilinear", align_corners=True, antialias=True)
    mask = torch.sigmoid(mask_logits)
    mask = (mask > threshold)

    return mask


def nonwarp_mask(model, c, depth, divergence, convergence, mapper, threshold=0.15, dilation=2):
    # warp depth to the left
    depth3 = depth.repeat(1, 3, 1, 1)
    warped_depth = apply_divergence_nn_delta_weight(
        model, depth3, depth, divergence=divergence, convergence=convergence, steps=1,
        mapper=mapper, shift=-1, preserve_screen_border=False, enable_amp=True,
        return_mask=False,
    )

    warped_depth = warped_depth.mean(dim=1, keepdim=True)
    # warp warped_depth to the right and back to original position
    dummy = torch.zeros_like(c)
    _, mask_logits = apply_divergence_nn_delta_weight(
        model, dummy, warped_depth, divergence=divergence, convergence=convergence, steps=1,
        mapper=mapper, shift=1, preserve_screen_border=False, enable_amp=True,
        return_mask=True,
    )
    mask = postprocess_hole_mask(mask_logits, c.shape[-2:], threshold=threshold, dilation=dilation)

    return c, mask


def _test_nonwarp_mask():
    import torchvision.transforms.functional as TF
    import torchvision.io as io
    from .stereo_model_factory import create_stereo_model
    from . import models  # noqa

    model = create_stereo_model("mask_mlbw_l2", divergence=10, device_id=0)

    x = io.read_image("cc0/320/dog.png") / 255.0
    depth = io.read_image("cc0/depth/dog.png") / 65536.0
    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()

    x, mask = nonwarp_mask(model, x, depth, divergence=4 * 2, convergence=0, mapper="none", threshold=0.15, dilation=2)
    x = x.mean(dim=1, keepdim=True)
    x = torch.cat([x, mask, torch.zeros_like(mask)], dim=1)[0]
    TF.to_pil_image(x).show()


if __name__ == "__main__":
    _test_nonwarp_mask()
