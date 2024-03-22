import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import threading
import math
from tqdm import tqdm
from PIL import ImageDraw
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.models import load_model, compile_model
import nunif.utils.video as VU
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir
from nunif.device import create_device, autocast


FLOW_MODEL_PATH = path.join(path.dirname(__file__), "pretrained_models", "row_flow_v2.pth")
REMBG_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "rembg")
os.environ["U2NET_HOME"] = path.abspath(path.normpath(REMBG_MODEL_DIR))


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


def softplus01(depth, c=6):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = torch.log(1. + torch.exp(depth * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)


def distance_to_dispary(x, c):
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
        param = {
            # none 1x
            "mul_1": 4,    # smooth 1.5x
            "mul_2": 6,    # smooth 2x
            "mul_3": 8.4,  # smooth 3x
        }[name]
        return lambda x: softplus01(x, param)
    elif name in {"div_6", "div_4", "div_2", "div_1"}:
        # for ZoeDepth
        param = {
            "div_6": 0.6,
            "div_4": 0.4,
            "div_2": 0.2,
            "div_1": 0.1,
        }[name]
        return lambda x: distance_to_dispary(x, param)
    else:
        raise NotImplementedError()


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


def apply_divergence_grid_sample(c, depth, divergence, convergence, shift):
    # BCHW
    B, _, H, W = depth.shape
    shift_size = (-shift * divergence * 0.01)
    index_shift = depth * shift_size - (shift_size * convergence)
    mesh_y, mesh_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=c.device),
        torch.linspace(-1, 1, W, device=c.device),
        indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, H, W).expand(B, 1, H, W)
    mesh_x = mesh_x.reshape(1, 1, H, W).expand(B, 1, H, W)
    mesh_x = mesh_x - index_shift
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    if c.shape[2] != H or c.shape[3] != W:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    z = F.grid_sample(c, grid,
                      mode="bicubic", padding_mode="border", align_corners=True)
    z = torch.clamp(z, 0., 1.)
    return z


def apply_divergence_nn_LR(model, c, depth, divergence, convergence,
                           mapper, enable_amp):
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
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, H, device=c.device),
                                    torch.linspace(-1, 1, W, device=c.device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, H, W).expand(B, 1, H, W)
    mesh_x = mesh_x.reshape(1, 1, H, W).expand(B, 1, H, W)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    delta_scale = 1.0 / (W // 2 - 1)
    grid = grid + delta * delta_scale
    if c.shape[2] != H or c.shape[3] != W:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    z = F.grid_sample(c, grid, mode="bicubic", padding_mode="border", align_corners=True)
    z = torch.clamp(z, 0, 1)

    if shift > 0:
        z = torch.flip(z, (3,))

    return z


def has_rembg_model(model_type):
    return path.exists(path.join(REMBG_MODEL_DIR, f"{model_type}.onnx"))


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF_Full_SBS"
HALF_SBS_SUFFIX = "_LR"
VR180_SUFFIX = "_180x180_LR"
ANAGLYPH_SUFFIX = "_redcyan"  # temporary


# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, video=False, vr180=False, half_sbs=False, anaglyph=None):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
    elif anaglyph is not None:
        auto_detect_suffix = ANAGLYPH_SUFFIX
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

    if args.method == "grid_sample":
        depth = get_mapper(args.mapper)(depth)
        left_eye = apply_divergence_grid_sample(
            im_org, depth,
            args.divergence, convergence=args.convergence, shift=-1)
        right_eye = apply_divergence_grid_sample(
            im_org, depth,
            args.divergence, convergence=args.convergence, shift=1)
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
        l = (0.75 * g_l + 0.25 * b_l) ** (1.0 / 1.6)
        anaglyph = torch.cat((l, g_r, b_r), dim=0)
    elif anaglyph_type == "dubois":
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

        def dot_clamp(x, vec):
            return (x * vec).sum(dim=0, keepdim=True).clamp(0, 1)

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
            dot_clamp(left_eye, l_mat[0]) + dot_clamp(right_eye, r_mat[0]),
            dot_clamp(left_eye, l_mat[1]) + dot_clamp(right_eye, r_mat[1]),
            dot_clamp(left_eye, l_mat[2]) + dot_clamp(right_eye, r_mat[2]),
        ], dim=0)
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
    mean_depth, std_depth = round(depth.mean().item(), 4), round(depth.std().item(), 4)
    depth = normalize_depth(depth, depth_min=depth_min, depth_max=depth_max)
    depth2 = get_mapper(args.mapper)(depth)
    out = torch.cat([depth, depth2], dim=2).cpu()
    out = TF.to_pil_image(out)
    gc = ImageDraw.Draw(out)
    gc.text((16, 16), f"min={depth_min}\nmax={depth_max}\nmean={mean_depth}\nstd={std_depth}", "gray")

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
            return debug_depth_image(depth, args)


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
                                     vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph))
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
        #side_model = compile_model(side_model, dynamic=True)
        pass

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), video=True,
                                 vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    make_parent_dir(output_filename)
    if ema_normalize:
        args.state["ema"].clear()

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
        return VU.to_frame(process_image(VU.to_tensor(frame), args, depth_model, side_model,
                                         return_tensor=True))
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
                                 vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph))
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
    parser.add_argument("--method", type=str, default="row_flow",
                        choices=["grid_sample", "row_flow"],
                        help="left-right divergence method")
    parser.add_argument("--divergence", "-d", type=float, default=2.0, choices=[Range(0.0, 6.0)],
                        help=("strength of 3D effect"))
    parser.add_argument("--convergence", "-c", type=float, default=0.5, choices=[Range(0.0, 1.0)],
                        help=("(normalized) distance of convergence plane(screen position)"))
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
    parser.add_argument("--tune", type=str, nargs="+", default=["zerolatency"],
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
    parser.add_argument("--mapper", type=str,
                        choices=["auto", "pow2", "softplus", "softplus2",
                                 "div_6", "div_4", "div_2", "div_1",
                                 "none", "mul_1", "mul_2", "mul_3"],
                        help=("(re-)mapper function for depth. "
                              "if auto, div_6 for ZoeDepth model, none for DepthAnything model. "
                              "directly using this option is deprecated. "
                              "use --foreground-scale instead."))
    parser.add_argument("--foreground-scale", type=int, choices=[0, 1, 2, 3], default=0,
                        help="foreground scaling level. 0 is disabled")
    parser.add_argument("--vr180", action="store_true",
                        help="output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true",
                        help="output in Half SBS")
    parser.add_argument("--anaglyph", type=str, nargs="?", default=None, const="dubois",
                        choices=["color", "gray", "half-color", "wimmer", "wimmer2", "dubois"],
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
    parser.add_argument("--edge-dilation", type=int, default=2,
                        help="loop count of edge dilation. only used for DepthAnything model")
    parser.add_argument("--max-workers", type=int, default=0, choices=[0, 1, 2, 3, 4, 8, 16],
                        help="max inference worker threads for video processing. 0 is disabled")

    return parser


class EMAMinMax():
    def __init__(self, alpha=0.25):
        self.min = None
        self.max = None
        self.alpha = alpha

    def update(self, min_value, max_value):
        if self.min is None:
            self.min = float(min_value)
            self.max = float(max_value)
        else:
            self.min += (float(min_value) - self.min) * self.alpha
            self.max += (float(max_value) - self.max) * self.alpha

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
        "ema": EMAMinMax(),
        "device": create_device(args.gpu),
        "depth_utils": depth_utils
    }
    return args


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)
    assert not (args.half_sbs and args.vr180)

    if args.update:
        args.state["depth_utils"].force_update()

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

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
            args.mapper = ["none", "mul_1", "mul_2", "mul_3"][args.foreground_scale]
        elif args.state["depth_utils"].get_name() == "ZoeDepth":
            args.mapper = ["div_6", "div_4", "div_2", "div_1"][args.foreground_scale]

    if args.state["depth_model"] is not None:
        depth_model = args.state["depth_model"]
    else:
        depth_model = args.state["depth_utils"].load_model(model_type=args.depth_model, gpu=args.gpu,
                                                           height=args.zoed_height)
        args.state["depth_model"] = depth_model

    if args.method == "row_flow":
        side_model = load_model(FLOW_MODEL_PATH, device_ids=[args.gpu[0]])[0].eval()
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
                                     vr180=args.vr180, half_sbs=args.half_sbs, anaglyph=args.anaglyph))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb")
        output = process_image(im, args, depth_model, side_model)
        make_parent_dir(output_filename)
        output.save(output_filename)

    return args
