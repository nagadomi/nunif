import os
from os import path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import threading
import math
from tqdm import tqdm
from PIL import ImageDraw, Image
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.models import load_model  # , compile_model
import nunif.utils.video as VU
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir, TorchHubDir
from nunif.device import create_device, autocast, device_is_mps
from . import export_config
from . dilation import dilate_edge
from . forward_warp import apply_divergence_forward_warp


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
REMBG_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "rembg")
os.environ["U2NET_HOME"] = path.abspath(path.normpath(REMBG_MODEL_DIR))

ROW_FLOW_V2_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v2_20240130.pth"
ROW_FLOW_V3_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_20240417.pth"
ROW_FLOW_V3_SYM_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_sym_20240418.pth"


def normalize_depth(depth, depth_min=None, depth_max=None):
    depth = depth.float()
    if depth_min is None:
        depth_min = depth.min()
        depth_max = depth.max()

    if depth_max - depth_min > 0:
        depth = 1. - ((depth - depth_min) / (depth_max - depth_min))
    else:
        depth = torch.zeros_like(depth)
    return torch.clamp(depth, 0., 1.)


def convert_normalized_depth_to_uint16_numpy(depth):
    uint16_max = 0xffff
    depth = uint16_max * depth
    depth = depth.to(torch.int16).numpy().astype(np.uint16)
    return depth


def make_divergence_feature_value(divergence, convergence, image_width):
    # assert image_width <= 2048
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0

    return divergence_feature_value, convergence_feature_value


def make_input_tensor(c, depth, divergence, convergence,
                      image_width, depth_min=None, depth_max=None,
                      mapper="pow2", normalize=True):
    if normalize:
        depth = normalize_depth(depth.squeeze(0), depth_min, depth_max)
    else:
        depth = depth.squeeze(0)  # CHW -> HW
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = torch.full_like(depth, divergence_value, device=depth.device)
    convergence_feat = torch.full_like(depth, convergence_value, device=depth.device)

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


def softplus01(x, bias, scale):
    # x: 0-1 normalized
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = torch.log(1. + torch.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def inv_softplus01(x, bias, scale):
    min_v = ((torch.zeros(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    max_v = ((torch.ones(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    v = ((x - bias) * scale).expm1().clamp(min=1e-6).log()
    return (v - min_v) / (max_v - min_v)


def distance_to_disparity(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def get_mapper(name):
    # https://github.com/nagadomi/nunif/assets/287255/0071a65a-62ff-4928-850c-0ad22bceba41
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01
    elif name == "softplus2":
        return lambda x: softplus01(x) ** 2
    elif name in {"mul_1", "mul_2", "mul_3"}:
        # for DepthAnything
        # https://github.com/nagadomi/nunif/assets/287255/2be5c0de-cb72-4c9c-9e95-4855c0730e5c
        param = {
            # none 1x
            "mul_1": {"bias": 0.343, "scale": 12},  # smooth 1.5x
            "mul_2": {"bias": 0.515, "scale": 12},  # smooth 2x
            "mul_3": {"bias": 0.687, "scale": 12},  # smooth 3x
        }[name]
        return lambda x: softplus01(x, **param)
    elif name in {"inv_mul_1", "inv_mul_2", "inv_mul_3"}:
        # for DepthAnything
        # https://github.com/nagadomi/nunif/assets/287255/f580b405-b0bf-4c6a-8362-66372b2ed930
        param = {
            # none 1x
            "inv_mul_1": {"bias": -0.002102, "scale": 7.8788},  # inverse smooth 1.5x
            "inv_mul_2": {"bias": -0.0003, "scale": 6.2626},    # inverse smooth 2x
            "inv_mul_3": {"bias": -0.0001, "scale": 3.4343},    # inverse smooth 3x
        }[name]
        return lambda x: inv_softplus01(x, **param)
    elif name in {"div_25", "div_10", "div_6", "div_4", "div_2", "div_1"}:
        # for ZoeDepth
        # TODO: There is no good reason for this parameter step
        # https://github.com/nagadomi/nunif/assets/287255/46c6b292-040f-4820-93fc-9e001cd53375
        param = {
            "div_25": 2.5,
            "div_10": 1,
            "div_6": 0.6,
            "div_4": 0.4,
            "div_2": 0.2,
            "div_1": 0.1,
        }[name]
        return lambda x: distance_to_disparity(x, param)
    else:
        raise NotImplementedError(f"mapper={name}")


def equirectangular_projection(c, device="cpu"):
    c = c.to(device)
    h, w = c.shape[1:]
    max_edge = max(h, w)
    output_size = max_edge + max_edge // 2
    pad_w = (output_size - w) // 2
    pad_h = (output_size - h) // 2
    c = TF.pad(c, (pad_w, pad_h, pad_w, pad_h),
               padding_mode="constant", fill=0)

    h, w = c.shape[1:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                          torch.linspace(-1, 1, w, device=device), indexing="ij")

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = (max_edge / output_size) * torch.tan(azimuth)
    mesh_y = (max_edge / output_size) * (torch.tan(elevation) / torch.cos(azimuth))
    grid = torch.stack((mesh_x, mesh_y), 2)
    z = F.grid_sample(c.unsqueeze(0),
                      grid.unsqueeze(0),
                      mode="bicubic", padding_mode="zeros",
                      align_corners=True).squeeze(0)
    z = torch.clamp(z, 0, 1)

    return z


def backward_warp(c, grid, delta, delta_scale):
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[2]:
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
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                    torch.linspace(-1, 1, width, device=device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).expand(batch, 1, height, width)
    mesh_x = mesh_x.reshape(1, 1, height, width).expand(batch, 1, height, width)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    return grid


def apply_divergence_grid_sample(c, depth, divergence, convergence):
    # BCHW
    B, _, H, W = depth.shape
    shift_size = divergence * 0.01
    index_shift = depth * shift_size - (shift_size * convergence)
    delta = torch.cat([index_shift, torch.zeros_like(index_shift)], dim=1)
    grid = make_grid(B, W, H, c.device)
    left_eye = backward_warp(c, grid, -delta, 1)
    right_eye = backward_warp(c, grid, delta, 1)

    return left_eye, right_eye


def apply_divergence_nn_LR(model, c, depth, divergence, convergence,
                           mapper, enable_amp):
    if getattr(model, "symmetric", False):
        left_eye, right_eye = apply_divergence_nn_symmetric(model, c, depth, divergence, convergence, mapper, enable_amp)
    else:
        left_eye = apply_divergence_nn(model, c, depth, divergence, convergence,
                                       mapper, -1, enable_amp)
        right_eye = apply_divergence_nn(model, c, depth, divergence, convergence,
                                        mapper, 1, enable_amp)
    return left_eye, right_eye


def apply_divergence_nn(model, c, depth, divergence, convergence,
                        mapper, shift, enable_amp):
    # BCHW
    assert model.delta_output
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    x = torch.stack([make_input_tensor(None, depth[i],
                                       divergence=divergence,
                                       convergence=convergence,
                                       image_width=W,
                                       mapper=mapper,
                                       normalize=False)  # already normalized
                     for i in range(depth.shape[0])])
    with autocast(device=depth.device, enabled=enable_amp):
        delta = model(x)
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    z = backward_warp(c, grid, delta, delta_scale)

    if shift > 0:
        z = torch.flip(z, (3,))

    return z


def apply_divergence_nn_symmetric(model, c, depth, divergence, convergence,
                                  mapper, enable_amp):
    # BCHW
    assert model.delta_output
    assert model.symmetric
    B, _, H, W = depth.shape

    x = torch.stack([make_input_tensor(None, depth[i],
                                       divergence=divergence,
                                       convergence=convergence,
                                       image_width=W,
                                       mapper=mapper,
                                       normalize=False)  # already normalized
                     for i in range(depth.shape[0])])
    with autocast(device=depth.device, enabled=enable_amp):
        delta = model(x)
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    left_eye = backward_warp(c, grid, delta, delta_scale)
    right_eye = backward_warp(c, grid, -delta, delta_scale)

    return left_eye, right_eye


def has_rembg_model(model_type):
    return path.exists(path.join(REMBG_MODEL_DIR, f"{model_type}.onnx"))


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF_Full_SBS"
HALF_SBS_SUFFIX = "_LR"
VR180_SUFFIX = "_180x180_LR"
ANAGLYPH_SUFFIX = "_redcyan"
DEBUG_SUFFIX = "_debug"

# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, video=False, vr180=False, half_sbs=False, anaglyph=None, debug=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
    elif anaglyph:
        auto_detect_suffix = ANAGLYPH_SUFFIX + f"_{anaglyph}"
    elif debug:
        auto_detect_suffix = DEBUG_SUFFIX
    else:
        auto_detect_suffix = FULL_SBS_SUFFIX

    return basename + auto_detect_suffix + (".mp4" if video else ".png")


def save_image(im, output_filename):
    im.save(output_filename)


def remove_bg_from_image(im, bg_session):
    # TODO: mask resolution seems to be low
    mask = TF.to_tensor(rembg.remove(im, session=bg_session, only_mask=True))
    im = TF.to_tensor(im)
    bg_color = torch.tensor((0.4, 0.4, 0.2)).view(3, 1, 1)
    im = im * mask + bg_color * (1.0 - mask)
    im = torch.clamp(im, 0, 1)
    im = TF.to_pil_image(im)

    return im


def preprocess_image(im, args):
    if not torch.is_tensor(im):
        im = TF.to_tensor(im)

    if args.rotate_left:
        im = torch.rot90(im, 1, (1, 2))
    elif args.rotate_right:
        im = torch.rot90(im, 3, (1, 2))

    h, w = im.shape[1:]
    new_w, new_h = w, h
    if args.max_output_height is not None and new_h > args.max_output_height:
        new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height
        # only apply max height
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        im = TF.resize(im, (new_h, new_w),
                       interpolation=InterpolationMode.BICUBIC, antialias=True)
        im = torch.clamp(im, 0, 1)
    im_org = im
    if args.bg_session is not None:
        im2 = remove_bg_from_image(TF.to_pil_image(im), args.bg_session)
        im = TF.to_tensor(im2).to(im.device)
    return im_org, im


def apply_divergence(depth, im_org, args, side_model, ema=False):
    batch = True
    if depth.ndim != 4:
        # CHW
        depth = depth.unsqueeze(0)
        im_org = im_org.unsqueeze(0)
        batch = False
    else:
        # BCHW
        pass

    for i in range(depth.shape[0]):
        depth_min, depth_max = depth[i].min(), depth[i].max()
        if ema:
            depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)
        depth[i] = normalize_depth(depth[i], depth_min=depth_min, depth_max=depth_max)

    if args.method in {"grid_sample", "backward"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_grid_sample(
            im_org, depth,
            args.divergence, convergence=args.convergence)
    elif args.method in {"forward", "forward_fill"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_forward_warp(
            im_org, depth,
            args.divergence, convergence=args.convergence,
            method=args.method)
    else:
        left_eye, right_eye = apply_divergence_nn_LR(
            side_model, im_org, depth,
            args.divergence, args.convergence,
            mapper=args.mapper,
            enable_amp=not args.disable_amp)

    if not batch:
        left_eye = left_eye.squeeze(0)
        right_eye = right_eye.squeeze(0)

    return left_eye, right_eye


def apply_anaglyph_redcyan(left_eye, right_eye, anaglyph_type):
    def grayscale_bt601(x, num_output_channels=1):
        y = x[0:1] * 0.299 + x[1:2] * 0.587 + x[2:3] * 0.114
        return torch.cat([y for _ in range(num_output_channels)], dim=0)

    if anaglyph_type == "color":
        anaglyph = torch.cat((left_eye[0:1, :, :], right_eye[1:3, :, :]), dim=0)
    elif anaglyph_type == "gray":
        ly = grayscale_bt601(left_eye, num_output_channels=3)
        ry = grayscale_bt601(right_eye, num_output_channels=3)
        anaglyph = torch.cat((ly[0:1, :, :], ry[1:3, :, :]), dim=0)
    elif anaglyph_type == "half-color":
        anaglyph = torch.cat((grayscale_bt601(left_eye, num_output_channels=1),
                              right_eye[1:3, :, :]), dim=0)
    elif anaglyph_type == "wimmer":
        # Wimmer's Optimized Anaglyph
        # https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx
        anaglyph = torch.cat((left_eye[1:2, :, :] * 0.7 + left_eye[2:3, :, :] * 0.3,
                              right_eye[1:3, :, :]), dim=0)
    elif anaglyph_type == "wimmer2":
        # Wimmer's improved method
        # described in "Methods for computing color anaglyphs"
        g_l = left_eye[1:2] + 0.45 * torch.clamp(left_eye[0:1] - left_eye[1:2], min=0)
        b_l = left_eye[2:3] + 0.25 * torch.clamp(left_eye[0:1] - left_eye[2:3], min=0)
        g_r = right_eye[1:2] + 0.45 * torch.clamp(right_eye[0:1] - right_eye[1:2], min=0)
        b_r = right_eye[2:3] + 0.25 * torch.clamp(right_eye[0:1] - right_eye[2:3], min=0)
        left = (0.75 * g_l + 0.25 * b_l) ** (1.0 / 1.6)
        anaglyph = torch.cat((left, g_r, b_r), dim=0)
    elif anaglyph_type in {"dubois", "dubois2"}:
        # Dubois method
        # reference: https://www.site.uottawa.ca/~edubois/anaglyph/LeastSquaresHowToPhotoshop.pdf
        def to_linear(x):
            cond1 = x <= 0.04045
            cond2 = torch.logical_not(cond1)
            x[cond1] = x[cond1] / 12.92
            x[cond2] = ((x[cond2] + 0.055) / 1.055) ** 2.4
            return x

        def to_nonlinear(x):
            cond1 = x <= 0.0031308
            cond2 = torch.logical_not(cond1)
            x[cond1] = x[cond1] * 12.92
            x[cond2] = 1.055 * x[cond2] ** (1.0 / 2.4) - 0.055
            return x

        def dot_clip(x, vec, clip):
            x = (x * vec).sum(dim=0, keepdim=True)
            if clip:
                x = x.clamp(0, 1)
            return x
        clip_before = True if anaglyph_type == "dubois" else False
        left_eye = to_linear(left_eye.detach().clone())
        right_eye = to_linear(right_eye.detach().clone())
        l_mat = torch.tensor([[0.437, 0.449, 0.164],
                              [-0.062, -0.062, -0.024],
                              [-0.048, -0.050, -0.017]],
                             device=left_eye.device, dtype=torch.float32).reshape(3, 3, 1, 1)
        r_mat = torch.tensor([[-0.011, -0.032, -0.007],
                              [0.377, 0.761, 0.009],
                              [-0.026, -0.093, 1.234]],
                             device=right_eye.device, dtype=torch.float32).reshape(3, 3, 1, 1)
        anaglyph = torch.cat([
            dot_clip(left_eye, l_mat[0], clip_before) + dot_clip(right_eye, r_mat[0], clip_before),
            dot_clip(left_eye, l_mat[1], clip_before) + dot_clip(right_eye, r_mat[1], clip_before),
            dot_clip(left_eye, l_mat[2], clip_before) + dot_clip(right_eye, r_mat[2], clip_before),
        ], dim=0)
        anaglyph = torch.clamp(anaglyph, 0, 1)
        anaglyph = to_nonlinear(anaglyph)

    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def postprocess_image(left_eye, right_eye, args):
    # CHW
    ipd_pad = int(abs(args.ipd_offset) * 0.01 * left_eye.shape[2])
    ipd_pad -= ipd_pad % 2
    if ipd_pad > 0:
        pad_o, pad_i = (ipd_pad * 2, ipd_pad) if args.ipd_offset > 0 else (ipd_pad, ipd_pad * 2)
        left_eye = TF.pad(left_eye, (pad_o, 0, pad_i, 0), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_i, 0, pad_o, 0), padding_mode="constant")

    if args.pad is not None:
        pad_h = int(left_eye.shape[1] * args.pad) // 2
        pad_w = int(left_eye.shape[2] * args.pad) // 2
        left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
    if args.vr180:
        left_eye = equirectangular_projection(left_eye, device=args.state["device"])
        right_eye = equirectangular_projection(right_eye, device=args.state["device"])
    elif args.half_sbs:
        left_eye = TF.resize(left_eye, (left_eye.shape[1], left_eye.shape[2] // 2),
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
        right_eye = TF.resize(right_eye, (right_eye.shape[1], right_eye.shape[2] // 2),
                              interpolation=InterpolationMode.BICUBIC, antialias=True)

    if args.anaglyph is None:
        sbs = torch.cat([left_eye, right_eye], dim=2)
        sbs = torch.clamp(sbs, 0., 1.)
    else:
        sbs = apply_anaglyph_redcyan(left_eye, right_eye, args.anaglyph)

    h, w = sbs.shape[1:]
    new_w, new_h = w, h
    if args.max_output_height is not None and new_h > args.max_output_height:
        if args.keep_aspect_ratio:
            new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height
    if args.max_output_width is not None and new_w > args.max_output_width:
        if args.keep_aspect_ratio:
            new_h = int(args.max_output_width / new_w * new_h)
        new_w = args.max_output_width
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        sbs = TF.resize(sbs, (new_h, new_w),
                        interpolation=InterpolationMode.BICUBIC, antialias=True)
        sbs = torch.clamp(sbs, 0, 1)
    return sbs


def debug_depth_image(depth, args, ema=False):
    depth = depth.float()
    depth_min, depth_max = depth.min(), depth.max()
    if ema:
        depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)
    mean_depth, std_depth = depth.mean().item(), depth.std().item()
    depth = normalize_depth(depth, depth_min=depth_min, depth_max=depth_max)
    depth2 = get_mapper(args.mapper)(depth)
    out = torch.cat([depth, depth2], dim=2).cpu()
    out = TF.to_pil_image(out)
    gc = ImageDraw.Draw(out)
    gc.text((16, 16), (f"min={round(float(depth_min), 4)}\n"
                       f"max={round(float(depth_max), 4)}\n"
                       f"mean={round(float(mean_depth), 4)}\n"
                       f"std={round(float(std_depth), 4)}"), "gray")

    return out


def process_image(im, args, depth_model, side_model, return_tensor=False):
    with torch.inference_mode():
        im_org, im = preprocess_image(im, args)
        depth = args.state["depth_utils"].batch_infer(
            depth_model, im, flip_aug=args.tta, low_vram=args.low_vram,
            int16=False, enable_amp=not args.disable_amp,
            output_device=args.state["device"],
            device=args.state["device"],
            edge_dilation=args.edge_dilation,
            resize_depth=False)
        if not args.debug_depth:
            left_eye, right_eye = apply_divergence(depth, im_org.to(args.state["device"]),
                                                   args, side_model)
            sbs = postprocess_image(left_eye, right_eye, args)
            if not return_tensor:
                sbs = TF.to_pil_image(sbs)
            return sbs
        else:
            return debug_depth_image(depth, args, args.ema_normalize)


def process_images(files, output_dir, args, depth_model, side_model, title=None):
    os.makedirs(output_dir, exist_ok=True)
    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title)
    with PoolExecutor(max_workers=4) as pool:
        for im, meta in loader:
            filename = meta["filename"]
            output_filename = path.join(
                output_dir,
                make_output_filename(filename, video=False,
                                     vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                     debug=args.debug_depth))
            if im is None or (args.resume and path.exists(output_filename)):
                continue
            output = process_image(im, args, depth_model, side_model)
            f = pool.submit(save_image, output, output_filename)
            #  f.result() # for debug
            futures.append(f)
            pbar.update(1)
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                break
        for f in futures:
            f.result()
    pbar.close()


def process_video_full(input_filename, output_path, args, depth_model, side_model):
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if side_model is not None:
        # TODO: sometimes ERROR RUNNING GUARDS forward error happen
        # side_model = compile_model(side_model, dynamic=True)
        pass

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), video=True,
                                 vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                 debug=args.debug_depth))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    make_parent_dir(output_filename)

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps

        options = {"preset": args.preset, "crf": str(args.crf), "frame-packing": "3"}
        tune = []
        if args.tune:
            tune += args.tune
        tune = set(tune)
        if tune:
            options["tune"] = ",".join(tune)
        return VU.VideoOutputConfig(
            fps=fps,
            pix_fmt=args.pix_fmt,
            options=options,
            container_options={"movflags": "+faststart"}
        )

    @torch.inference_mode()
    def test_callback(frame):
        frame = VU.to_frame(process_image(VU.to_tensor(frame), args, depth_model, side_model,
                                          return_tensor=True))
        if ema_normalize:
            args.state["ema"].clear()
        return frame

    if args.low_vram or args.debug_depth:
        @torch.inference_mode()
        def frame_callback(frame):
            if frame is None:
                return None
            return VU.to_frame(process_image(VU.to_tensor(frame), args, depth_model, side_model,
                                             return_tensor=True))

        VU.process_video(input_filename, output_filename,
                         config_callback=config_callback,
                         frame_callback=frame_callback,
                         test_callback=test_callback,
                         vf=args.vf,
                         stop_event=args.state["stop_event"],
                         tqdm_fn=args.state["tqdm_fn"],
                         title=path.basename(input_filename),
                         start_time=args.start_time,
                         end_time=args.end_time)
    else:
        minibatch_size = args.zoed_batch_size // 2 or 1 if args.tta else args.zoed_batch_size
        preprocess_lock = threading.Lock()
        depth_lock = threading.Lock()
        sbs_lock = threading.Lock()

        @torch.inference_mode()
        def _batch_callback(x):
            if args.max_output_height is not None or args.bg_session is not None:
                # TODO: batch preprocess_image
                with preprocess_lock:
                    xs = [preprocess_image(xx, args) for xx in x]
                    x = torch.stack([x for x_org, x in xs])
                    if args.bg_session is not None:
                        x_orgs = torch.stack([x_org for x_org, x in xs])
                    else:
                        x_orgs = x
            else:
                x_orgs = x
            with depth_lock:
                depths = args.state["depth_utils"].batch_infer(
                    depth_model, x, flip_aug=args.tta, low_vram=args.low_vram,
                    int16=False, enable_amp=not args.disable_amp,
                    output_device=args.state["device"],
                    device=args.state["device"],
                    edge_dilation=args.edge_dilation,
                    resize_depth=False)
            if args.method in {"forward", "forward_fill"}:
                # Lock all threads
                # forward_warp uses torch.use_deterministic_algorithms() and it seems to be not thread-safe
                with sbs_lock, preprocess_lock, depth_lock:
                    left_eyes, right_eyes = apply_divergence(depths, x_orgs, args, side_model, ema_normalize)
            else:
                with sbs_lock:
                    left_eyes, right_eyes = apply_divergence(depths, x_orgs, args, side_model, ema_normalize)

            return torch.stack([
                postprocess_image(left_eyes[i], right_eyes[i], args)
                for i in range(left_eyes.shape[0])])
        frame_callback = VU.FrameCallbackPool(
            _batch_callback,
            batch_size=minibatch_size,
            device=args.state["device"],
            max_workers=args.max_workers,
            max_batch_queue=args.max_workers + 1,
        )
        VU.process_video(input_filename, output_filename,
                         config_callback=config_callback,
                         frame_callback=frame_callback,
                         test_callback=test_callback,
                         vf=args.vf,
                         stop_event=args.state["stop_event"],
                         tqdm_fn=args.state["tqdm_fn"],
                         title=path.basename(input_filename),
                         start_time=args.start_time,
                         end_time=args.end_time)
        frame_callback.shutdown()


def process_video_keyframes(input_filename, output_path, args, depth_model, side_model):
    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), video=True,
                                 vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                 debug=args.debug_depth))
    else:
        output_filename = output_path

    output_dir = path.join(path.dirname(output_filename), path.splitext(path.basename(output_filename))[0])
    if output_dir.endswith("_LRF"):
        output_dir = output_dir[:-4]
    os.makedirs(output_dir, exist_ok=True)
    with PoolExecutor(max_workers=4) as pool:
        futures = []

        def frame_callback(frame):
            output = process_image(frame.to_image(), args, depth_model, side_model)
            output_filename = path.join(
                output_dir,
                path.basename(output_dir) + "_" + str(frame.index).zfill(8) + FULL_SBS_SUFFIX + ".png")
            f = pool.submit(save_image, output, output_filename)
            futures.append(f)
        VU.process_video_keyframes(input_filename, frame_callback=frame_callback,
                                   min_interval_sec=args.keyframe_interval,
                                   stop_event=args.state["stop_event"],
                                   title=path.basename(input_filename))
        for f in futures:
            f.result()


def process_video(input_filename, output_path, args, depth_model, side_model):
    if args.keyframe:
        process_video_keyframes(input_filename, output_path, args, depth_model, side_model)
    else:
        process_video_full(input_filename, output_path, args, depth_model, side_model)


def export_images(files, args):
    if args.export_disparity:
        mapper = "none"
        edge_dilation = args.edge_dilation
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        edge_dilation = 0
        skip_edge_dilation = False
        skip_mapper = False
    config = export_config.ExportConfig(
        type=export_config.IMAGE_TYPE,
        fps=1,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "ema_normalize": False,
            }
        }
    )
    config.audio_file = None
    output_dir = args.output
    rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    config_file = path.join(output_dir, export_config.FILENAME)

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    depth_model = args.state["depth_model"]

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc="Images")
    with PoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for im, meta in loader:
            basename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            rgb_file = path.join(rgb_dir, basename)
            depth_file = path.join(depth_dir, basename)
            if im is None or (args.resume and path.exists(rgb_file) and path.exists(rgb_file)):
                continue

            im_org, im = preprocess_image(im, args)
            depth = args.state["depth_utils"].batch_infer(
                depth_model, im, flip_aug=args.tta, low_vram=args.low_vram,
                int16=False, enable_amp=not args.disable_amp,
                output_device=args.state["device"],
                device=args.state["device"],
                edge_dilation=edge_dilation,
                resize_depth=False)

            depth = normalize_depth(depth)
            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            depth = convert_normalized_depth_to_uint16_numpy(depth.detach().cpu()[0])
            depth = Image.fromarray(depth)
            im_org = TF.to_pil_image(im_org)
            futures.append(pool.submit(save_image, depth, depth_file))
            futures.append(pool.submit(save_image, im_org, rgb_file))
            pbar.update(1)
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                break

        for f in futures:
            f.result()
    pbar.close()
    config.save(config_file)


def export_video(args):
    basename = path.splitext(path.basename(args.input))[0]
    if args.export_disparity:
        mapper = "none"
        edge_dilation = args.edge_dilation
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        edge_dilation = 0
        skip_edge_dilation = False
        skip_mapper = False
    config = export_config.ExportConfig(
        type=export_config.VIDEO_TYPE,
        basename=basename,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "max_fps": args.max_fps,
                "ema_normalize": args.ema_normalize,
            }
        }
    )
    # NOTE: Windows does not allow creating folders with trailing spaces. basename.strip()
    output_dir = path.join(args.output, basename.strip())
    rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    audio_file = path.join(output_dir, config.audio_file)
    config_file = path.join(output_dir, export_config.FILENAME)

    if not args.yes and path.exists(config_file):
        y = input(f"File '{config_file}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    has_audio = VU.export_audio(args.input, audio_file,
                                start_time=args.start_time, end_time=args.end_time,
                                title="Audio", stop_event=args.state["stop_event"],
                                tqdm_fn=args.state["tqdm_fn"])
    if not has_audio:
        config.audio_file = None

    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
        return

    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        args.state["ema"].clear()

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        config.fps = fps  # update fps
        return VU.VideoOutputConfig(fps=fps)

    minibatch_size = args.zoed_batch_size // 2 or 1 if args.tta else args.zoed_batch_size
    preprocess_lock = threading.Lock()
    depth_lock = threading.Lock()
    depth_model = args.state["depth_model"]

    @torch.inference_mode()
    def _batch_callback(x, pts):
        if args.max_output_height is not None or args.bg_session is not None:
            with preprocess_lock:
                xs = [preprocess_image(xx, args) for xx in x]
                x = torch.stack([x for x_org, x in xs])
                if args.bg_session is not None:
                    x_orgs = torch.stack([x_org for x_org, x in xs])
                else:
                    x_orgs = x
        else:
            x_orgs = x

        with depth_lock:
            depths = args.state["depth_utils"].batch_infer(
                depth_model, x,
                int16=False,
                flip_aug=args.tta, low_vram=args.low_vram,
                enable_amp=not args.disable_amp,
                output_device=args.state["device"],
                device=args.state["device"],
                edge_dilation=edge_dilation,
                resize_depth=False)

            for i in range(depths.shape[0]):
                depth_min, depth_max = depths[i].min(), depths[i].max()
                if ema_normalize:
                    depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)
                depth = normalize_depth(depths[i], depth_min=depth_min, depth_max=depth_max)
                if args.export_disparity:
                    depth = get_mapper(args.mapper)(depth)
                depths[i] = depth

        depths = depths.detach().cpu()
        x_orgs = x_orgs.detach().cpu()

        for x, depth, seq in zip(x_orgs, depths, pts):
            seq = str(seq).zfill(8)
            depth = convert_normalized_depth_to_uint16_numpy(depth[0])
            dpeth = Image.fromarray(depth)
            dpeth.save(path.join(depth_dir, f"{seq}.png"))
            rgb = TF.to_pil_image(x)
            rgb.save(path.join(rgb_dir, f"{seq}.png"))

    frame_callback = VU.FrameCallbackPool(
        _batch_callback,
        batch_size=minibatch_size,
        device=args.state["device"],
        max_workers=args.max_workers,
        max_batch_queue=args.max_workers + 1,
        require_pts=True,
    )
    VU.hook_frame(args.input,
                  config_callback=config_callback,
                  frame_callback=frame_callback,
                  vf=args.vf,
                  stop_event=args.state["stop_event"],
                  tqdm_fn=args.state["tqdm_fn"],
                  title=path.basename(args.input),
                  start_time=args.start_time,
                  end_time=args.end_time)
    frame_callback.shutdown()
    config.save(config_file)


def to_float32_grayscale_depth(depth):
    if depth.dtype == torch.int32:
        # 16bit image
        depth = torch.clamp(depth.to(torch.float32) / 0xffff, 0, 1)
    if depth.shape[0] != 1:
        # Maybe 24bpp
        # TODO: color depth support?
        depth = torch.mean(depth, dim=0, keepdim=True)
    # invert
    depth = 1. - depth
    return depth


def process_config_video(config, args, side_model):
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, audio_file = config.resolve_paths(base_dir)

    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        basename = config.basename or path.basename(base_dir)
        output_filename = path.join(
            args.output,
            make_output_filename(basename, video=True,
                                 vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                 debug=args.debug_depth))
    else:
        output_filename = args.output
    make_parent_dir(output_filename)

    if args.resume and path.exists(output_filename):
        return
    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "any"})

    sbs_lock = threading.Lock()

    @torch.inference_mode()
    def batch_callback(x, depths):
        if not config.skip_edge_dilation and args.edge_dilation > 0:
            # apply --edge-dilation
            depths = -dilate_edge(-depths, args.edge_dilation)
        with sbs_lock:
            left_eyes, right_eyes = apply_divergence(depths, x, args, side_model)
        return torch.stack([
            postprocess_image(left_eyes[i], right_eyes[i], args)
            for i in range(left_eyes.shape[0])])

    def test_output_size(rgb_file, depth_file):
        rgb = load_image_simple(rgb_file, color="rgb")[0]
        depth = load_image_simple(depth_file, color="any")[0]
        rgb = TF.to_tensor(rgb)
        depth = to_float32_grayscale_depth(TF.to_tensor(depth))
        frame = batch_callback(rgb.unsqueeze(0).to(args.state["device"]),
                               depth.unsqueeze(0).to(args.state["device"]))
        return frame.shape[2:]

    minibatch_size = args.zoed_batch_size // 2 or 1 if args.tta else args.zoed_batch_size

    def generator():
        rgb_batch = []
        depth_batch = []
        for rgb, depth in zip(rgb_loader, depth_loader):
            rgb = TF.to_tensor(rgb[0])
            depth = to_float32_grayscale_depth(TF.to_tensor(depth[0]))
            rgb_batch.append(rgb)
            depth_batch.append(depth)
            if len(rgb_batch) == minibatch_size:
                frames = batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                        torch.stack(depth_batch).to(args.state["device"]))
                rgb_batch.clear()
                depth_batch.clear()

                yield [VU.to_frame(frame) for frame in frames]

        if rgb_batch:
            frames = batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                    torch.stack(depth_batch).to(args.state["device"]))
            rgb_batch.clear()
            depth_batch.clear()

            yield [VU.to_frame(frame) for frame in frames]

    output_height, output_width = test_output_size(rgb_files[0], depth_files[0])
    encoder_options = {"preset": args.preset, "crf": str(args.crf), "frame-packing": "3"}
    if args.tune:
        encoder_options.update({"tune": ",".join(list(set(args.tune)))})
    video_config = VU.VideoOutputConfig(
        fps=config.fps,  # use config.fps, ignore args.max_fps
        pix_fmt=args.pix_fmt,
        options=encoder_options,
        container_options={"movflags": "+faststart"},
        output_width=output_width,
        output_height=output_height
    )
    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            # force use "none" mapper
            args.mapper = "none"
        else:
            if config.mapper is not None:
                # use specified mapper from config
                # NOTE: override args
                args.mapper = config.mapper
            else:
                # when config.mapper is not defined, use args.mapper
                # TODO: It can be still disputable
                pass
        VU.generate_video(
            output_filename,
            generator,
            config=video_config,
            audio_file=audio_file,
            title=path.basename(base_dir),
            total_frames=len(rgb_files),
            stop_event=args.state["stop_event"],
            tqdm_fn=args.state["tqdm_fn"],
        )
    finally:
        args.mapper = original_mapper


def process_config_images(config, args, side_model):
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, _ = config.resolve_paths(base_dir)
    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "any"})

    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            args.mapper = "none"
        else:
            if config.mapper is not None:
                args.mapper = config.mapper
            else:
                pass
        with PoolExecutor(max_workers=4) as pool:
            tqdm_fn = args.state["tqdm_fn"] or tqdm
            pbar = tqdm_fn(ncols=80, total=len(rgb_files), desc="Images")
            futures = []
            for (rgb, rgb_meta), (depth, depth_meta) in zip(rgb_loader, depth_loader):
                rgb_filename = path.splitext(path.basename(rgb_meta["filename"]))[0]
                depth_filename = path.splitext(path.basename(depth_meta["filename"]))[0]
                if rgb_filename != depth_filename:
                    raise ValueError(f"No match {rgb_filename} and {depth_filename}")
                rgb = TF.to_tensor(rgb)
                depth = to_float32_grayscale_depth(TF.to_tensor(depth))
                if not config.skip_edge_dilation and args.edge_dilation > 0:
                    depth = -dilate_edge(-depth.unsqueeze(0), args.edge_dilation).squeeze(0)

                left_eye, right_eye = apply_divergence(
                    depth.to(args.state["device"]),
                    rgb.to(args.state["device"]),
                    args, side_model)
                sbs = postprocess_image(left_eye, right_eye, args)
                sbs = TF.to_pil_image(sbs)

                output_filename = path.join(
                    output_dir,
                    make_output_filename(rgb_filename, video=False,
                                         vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                         debug=args.debug_depth))
                f = pool.submit(save_image, sbs, output_filename)
                futures.append(f)
                pbar.update(1)
            for f in futures:
                f.result()
            pbar.close()
    finally:
        args.mapper = original_mapper


def create_parser(required_true=True):
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

        def __repr__(self):
            return f"{self.start} <= value <= {self.end}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser.add_argument("--input", "-i", type=str, required=required_true,
                        help="input file or directory")
    parser.add_argument("--output", "-o", type=str, required=required_true,
                        help="output file or directory")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--method", type=str, default="row_flow_sym",
                        choices=["grid_sample", "backward", "forward", "forward_fill",
                                 "row_flow", "row_flow_sym",
                                 "row_flow_v3", "row_flow_v3_sym",
                                 "row_flow_v2"],
                        help="left-right divergence method")
    parser.add_argument("--divergence", "-d", type=float, default=2.0,
                        help=("strength of 3D effect. 0-2 is reasonable value"))
    parser.add_argument("--convergence", "-c", type=float, default=0.5,
                        help=("(normalized) distance of convergence plane(screen position). 0-1 is reasonable value"))
    parser.add_argument("--update", action="store_true",
                        help="force update midas models from torch hub")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="process all subdirectories")
    parser.add_argument("--resume", action="store_true",
                        help="skip processing when the output file already exists")
    parser.add_argument("--batch-size", type=int, default=16, choices=[Range(1, 256)],
                        help="batch size for RowFlow model, 256x256 tiled input. !!DEPRECATED!!")
    parser.add_argument("--zoed-batch-size", type=int, default=2, choices=[Range(1, 64)],
                        help="batch size for ZoeDepth model. ignored when --low-vram")
    parser.add_argument("--max-fps", type=float, default=30,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value for video. smaller value is higher quality")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo"],
                        help="encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency"],
                        help="encoder tunings option for video")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="overwrite output files")
    parser.add_argument("--pad", type=float, help="pad_size = int(size * pad)")
    parser.add_argument("--depth-model", type=str, default="ZoeD_N",
                        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK",
                                 "Any_S", "Any_B", "Any_L",
                                 "ZoeD_Any_N", "ZoeD_Any_K"],
                        help="depth model name")
    parser.add_argument("--remove-bg", action="store_true",
                        help="remove background depth, not recommended for video")
    parser.add_argument("--bg-model", type=str, default="u2net_human_seg",
                        help="rembg model type")
    parser.add_argument("--rotate-left", action="store_true",
                        help="Rotate 90 degrees to the left(counterclockwise)")
    parser.add_argument("--rotate-right", action="store_true",
                        help="Rotate 90 degrees to the right(clockwise)")
    parser.add_argument("--low-vram", action="store_true",
                        help="disable batch processing for low memory GPU")
    parser.add_argument("--keyframe", action="store_true",
                        help="process only keyframe as image")
    parser.add_argument("--keyframe-interval", type=float, default=4.0,
                        help="keyframe minimum interval (sec)")
    parser.add_argument("--vf", type=str, default="",
                        help="video filter options for ffmpeg.")
    parser.add_argument("--debug-depth", action="store_true",
                        help="debug output normalized depthmap, info and preprocessed depth")
    parser.add_argument("--export", action="store_true", help="export depth, frame, audio")
    parser.add_argument("--export-disparity", action="store_true",
                        help=("export dispary instead of depth. "
                              "this means applying --mapper and --foreground-scale."))
    parser.add_argument("--mapper", type=str,
                        choices=["auto", "pow2", "softplus", "softplus2",
                                 "div_6", "div_4", "div_2", "div_1",
                                 "none", "mul_1", "mul_2", "mul_3",
                                 "inv_mul_1", "inv_mul_2", "inv_mul_3",
                                 ],
                        help=("(re-)mapper function for depth. "
                              "if auto, div_6 for ZoeDepth model, none for DepthAnything model. "
                              "directly using this option is deprecated. "
                              "use --foreground-scale instead."))
    parser.add_argument("--foreground-scale", type=int, choices=[-3, -2, -1, 0, 1, 2, 3], default=0,
                        help="foreground scaling level. 0 is disabled")
    parser.add_argument("--vr180", action="store_true",
                        help="output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true",
                        help="output in Half SBS")
    parser.add_argument("--anaglyph", type=str, nargs="?", default=None, const="dubois",
                        choices=["color", "gray", "half-color", "wimmer", "wimmer2", "dubois", "dubois2"],
                        help="output in anaglyph 3d")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p", choices=["yuv420p", "yuv444p", "rgb24"],
                        help="pixel format (video only)")
    parser.add_argument("--tta", action="store_true",
                        help="Use flip augmentation on depth model")
    parser.add_argument("--disable-amp", action="store_true",
                        help="disable AMP for some special reason")
    parser.add_argument("--max-output-width", type=int,
                        help="limit output width for cardboard players")
    parser.add_argument("--max-output-height", type=int,
                        help="limit output height for cardboard players")
    parser.add_argument("--keep-aspect-ratio", action="store_true",
                        help="keep aspect ratio when resizing")
    parser.add_argument("--start-time", type=str,
                        help="set the start time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--end-time", type=str,
                        help="set the end time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--zoed-height", type=int,
                        help="input height for ZoeDepth model")
    parser.add_argument("--ipd-offset", type=float, default=0,
                        help="IPD Offset (width scale %%). 0-10 is reasonable value for Full SBS")
    parser.add_argument("--ema-normalize", action="store_true",
                        help="use min/max moving average to normalize video depth")
    parser.add_argument("--ema-decay", type=float, default=0.75,
                        help="parameter for ema-normalize (0-1). large value makes it smoother")
    parser.add_argument("--edge-dilation", type=int, nargs="?", default=None, const=2,
                        help="loop count of edge dilation.")
    parser.add_argument("--max-workers", type=int, default=0, choices=[0, 1, 2, 3, 4, 8, 16],
                        help="max inference worker threads for video processing. 0 is disabled")

    return parser


class EMAMinMax():
    def __init__(self, alpha=0.75):
        self.min = None
        self.max = None
        self.alpha = alpha

    def update(self, min_value, max_value):
        if self.min is None:
            self.min = float(min_value)
            self.max = float(max_value)
        else:
            self.min = self.alpha * self.min + (1. - self.alpha) * float(min_value)
            self.max = self.alpha * self.max + (1. - self.alpha) * float(max_value)

        # print(round(float(min_value), 3), round(float(max_value), 3), round(self.min, 3), round(self.max, 3))

        return self.min, self.max

    def clear(self):
        self.min = self.max = None


def set_state_args(args, stop_event=None, tqdm_fn=None, depth_model=None):
    from . import zoedepth_model as ZU
    from . import depth_anything_model as DU
    if args.depth_model in ZU.MODEL_FILES:
        depth_utils = ZU
    elif args.depth_model in DU.MODEL_FILES:
        depth_utils = DU

    args.state = {
        "stop_event": stop_event,
        "tqdm_fn": tqdm_fn,
        "depth_model": depth_model,
        "ema": EMAMinMax(alpha=args.ema_decay),
        "device": create_device(args.gpu),
        "depth_utils": depth_utils
    }
    return args


def export_main(args):
    if args.recursive:
        raise NotImplementedError("`--recursive --export` is not supported")
    if is_text(args.input):
        raise NotImplementedError("--export with text format input is not supported")

    if path.isdir(args.input):
        image_files = ImageLoader.listdir(args.input)
        export_images(image_files, args)
    elif is_image(args.input):
        export_images([args.input], args)
    elif is_video(args.input):
        export_video(args)


def is_yaml(filename):
    return path.splitext(filename)[-1].lower() in {".yaml", ".yml"}


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)
    assert not (args.half_sbs and args.vr180)
    assert not (args.half_sbs and args.anaglyph)
    assert not (args.vr180 and args.anaglyph)

    if args.update:
        args.state["depth_utils"].force_update()

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

    if args.export_disparity:
        args.export = True

    if args.export and is_yaml(args.input):
        raise ValueError("YAML file input does not support --export")

    if args.remove_bg:
        global rembg
        import rembg
        args.bg_session = rembg.new_session(model_name=args.bg_model)
    else:
        args.bg_session = None

    if args.mapper is not None:
        if args.mapper == "auto":
            if args.state["depth_utils"].get_name() == "DepthAnything":
                args.mapper = "none"
            else:
                args.mapper = "div_6"
        else:
            pass
    else:
        if args.state["depth_utils"].get_name() == "DepthAnything":
            args.mapper = [
                "inv_mul_3", "inv_mul_2", "inv_mul_1",
                "none",
                "mul_1", "mul_2", "mul_3",
            ][args.foreground_scale + 3]
        elif args.state["depth_utils"].get_name() == "ZoeDepth":
            args.mapper = [
                "none", "div_25", "div_10",
                "div_6",
                "div_4", "div_2", "div_1",
            ][args.foreground_scale + 3]
    if args.edge_dilation is None:
        if args.state["depth_utils"].get_name() == "DepthAnything":
            args.edge_dilation = 2
        else:
            args.edge_dilation = 0

    if not is_yaml(args.input):
        if args.state["depth_model"] is not None:
            depth_model = args.state["depth_model"]
        else:
            depth_model = args.state["depth_utils"].load_model(model_type=args.depth_model, gpu=args.gpu,
                                                               height=args.zoed_height)
            args.state["depth_model"] = depth_model
    else:
        depth_model = None

    if args.export:
        export_main(args)
        return args

    with TorchHubDir(HUB_MODEL_DIR):
        if args.method in {"row_flow_v3_sym", "row_flow_sym"}:
            side_model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = True
            side_model.delta_output = True
        elif args.method in {"row_flow_v3", "row_flow"}:
            side_model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = False
            side_model.delta_output = True
        elif args.method == "row_flow_v2":
            side_model = load_model(ROW_FLOW_V2_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.delta_output = True
        else:
            side_model = None

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            image_files = ImageLoader.listdir(args.input)
            process_images(image_files, args.output, args, depth_model, side_model, title="Images")
            for video_file in VU.list_videos(args.input):
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return args
                process_video(video_file, args.output, args, depth_model, side_model)
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                image_files = ImageLoader.listdir(input_dir)
                if image_files:
                    process_images(image_files, output_dir, args, depth_model, side_model,
                                   title=path.relpath(input_dir, args.input))
                for video_file in VU.list_videos(input_dir):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    process_video(video_file, output_dir, args, depth_model, side_model)

    elif is_text(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        files = []
        with open(args.input, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                files.append(line.strip())
        image_files = [f for f in files if is_image(f)]
        process_images(image_files, args.output, args, depth_model, side_model, title="Images")
        video_files = [f for f in files if is_video(f)]
        for video_file in video_files:
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                return args
            process_video(video_file, args.output, args, depth_model, side_model)
    elif is_video(args.input):
        process_video(args.input, args.output, args, depth_model, side_model)
    elif is_image(args.input):
        if is_output_dir(args.output):
            os.makedirs(args.output, exist_ok=True)
            output_filename = path.join(
                args.output,
                make_output_filename(args.input, video=False,
                                     vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph,
                                     debug=args.debug_depth))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb")
        output = process_image(im, args, depth_model, side_model)
        make_parent_dir(output_filename)
        output.save(output_filename)
    elif is_yaml(args.input):
        config = export_config.ExportConfig.load(args.input)
        if config.type == export_config.VIDEO_TYPE:
            process_config_video(config, args, side_model)
        if config.type == export_config.IMAGE_TYPE:
            process_config_images(config, args, side_model)
    else:
        raise ValueError("Unrecognized file type")

    return args
