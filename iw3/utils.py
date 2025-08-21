import os
from os import path
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import threading
import math
from tqdm import tqdm
from PIL import Image
from nunif.initializer import gc_collect
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
import nunif.utils.shot_boundary_detection as SBD
from nunif.models import load_model, compile_model
import nunif.utils.video as VU
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir, TorchHubDir
from nunif.utils.ticket_lock import TicketLock
from nunif.device import create_device, device_is_cuda, mps_is_available, xpu_is_available
from nunif.models.data_parallel import DeviceSwitchInference
from . import export_config
from .dilation import dilate_edge
from .forward_warp import apply_divergence_forward_warp
from .anaglyph import apply_anaglyph_redcyan
from .mapper import get_mapper, resolve_mapper_name, MAPPER_ALL
from .depth_model_factory import create_depth_model
from .base_depth_model import BaseDepthModel
from .equirectangular import equirectangular_projection
from .backward_warp import (
    apply_divergence_grid_sample,
    apply_divergence_nn_LR,
)
from .stereo_model_factory import create_stereo_model


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
ROW_FLOW_V2_MAX_DIVERGENCE = 2.5
ROW_FLOW_V3_MAX_DIVERGENCE = 5.0
ROW_FLOW_V2_AUTO_STEP_DIVERGENCE = 2.0
ROW_FLOW_V3_AUTO_STEP_DIVERGENCE = 4.0
IMAGE_IO_QUEUE_MAX = 100


def chunks(array, n):
    for i in range(0, len(array), n):
        yield array[i:i + n]


def to_pil_image(x):
    # x is already clipped to 0-1
    assert x.dtype in {torch.float32, torch.float16}
    x = TF.to_pil_image((x * 255).round().to(torch.uint8).cpu())
    return x


def apply_rgbd(im, depth, mapper):
    height, width = im.shape[-2:]
    left_eye = im
    if mapper is not None:
        depth = get_mapper(mapper)(depth)

    if depth.ndim == 3:
        right_eye = F.interpolate(depth.unsqueeze(0), (height, width),
                                  mode="bicubic", antialias=True).squeeze(0)
    else:
        right_eye = F.interpolate(depth, (height, width),
                                  mode="bicubic", antialias=True)

    right_eye = right_eye.expand_as(left_eye)
    return left_eye, right_eye


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF_Full_SBS"
HALF_SBS_SUFFIX = "_LR"
FULL_TB_SUFFIX = "_TBF_fulltb"
HALF_TB_SUFFIX = "_TB"
CROSS_EYED_SUFFIX = "_RLF_cross"
RGBD_SUFFIX = "_RGBD"  # TODO
HALF_RGBD_SUFFIX = "_HRGBD"  # TODO

VR180_SUFFIX = "_180x180_LR"
ANAGLYPH_SUFFIX = "_redcyan"
DEBUG_SUFFIX = "_debug"

# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, args, video=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if args.vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif args.half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
    elif args.tb:
        auto_detect_suffix = FULL_TB_SUFFIX
    elif args.half_tb:
        auto_detect_suffix = HALF_TB_SUFFIX
    elif args.cross_eyed:
        auto_detect_suffix = CROSS_EYED_SUFFIX
    elif args.anaglyph:
        auto_detect_suffix = ANAGLYPH_SUFFIX + f"_{args.anaglyph}"
    elif args.rgbd:
        auto_detect_suffix = RGBD_SUFFIX
    elif args.half_rgbd:
        auto_detect_suffix = HALF_RGBD_SUFFIX
    elif args.debug_depth:
        auto_detect_suffix = DEBUG_SUFFIX
    else:
        auto_detect_suffix = FULL_SBS_SUFFIX

    def to_deciaml(f, scale, zfill=0):
        s = str(int(f * scale))
        if zfill:
            s = s.zfill(zfill)
        return s

    if args.metadata == "filename":
        if args.resolution:
            resolution = f"{args.resolution}_"
        else:
            resolution = ""
        if args.tta:
            tta = "TTA_"
        else:
            tta = ""
        if args.ema_normalize and video:
            ema = f"_ema{to_deciaml(args.ema_decay, 100, 2)}"
        else:
            ema = ""
        metadata = (f"_{args.depth_model}_{resolution}{tta}{args.method}_"
                    f"d{to_deciaml(args.divergence, 10, 2)}_c{to_deciaml(args.convergence, 10, 2)}_"
                    f"di{args.edge_dilation}_fs{args.foreground_scale}_ipd{to_deciaml(args.ipd_offset, 1)}{ema}")
    else:
        metadata = ""

    return basename + metadata + auto_detect_suffix + (args.video_extension if video else get_image_ext(args.format))


def make_video_codec_option(args):
    if args.video_codec in {"libx264", "libx265", "hevc_nvenc", "h264_nvenc"}:
        options = {"preset": args.preset, "crf": str(args.crf)}

        if args.tune:
            options["tune"] = ",".join(set(args.tune))

        if args.profile_level:
            options["level"] = str(int(float(args.profile_level) * 10))

        if args.video_codec == "libx265":
            x265_params = ["log-level=warning", "high-tier=enabled"]
            if args.profile_level:
                x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")
            options["x265-params"] = ":".join(x265_params)
        elif args.video_codec == "libx264":
            # TODO:
            # if args.tb or args.half_tb:
            #    options["x264-params"] = "frame-packing=4"
            if args.half_sbs:
                options["x264-params"] = "frame-packing=3"
        elif args.video_codec in {"hevc_nvenc", "h264_nvenc"}:
            options["rc"] = "constqp"
            options["qp"] = str(args.crf)
            if torch.cuda.is_available() and args.gpu[0] >= 0:
                options["gpu"] = str(args.gpu[0])
    elif args.video_codec == "libopenh264":
        # NOTE: It seems libopenh264 does not support most options.
        options = {"b": args.video_bitrate}
    else:
        options = {}

    return options


def get_image_ext(format):
    if format == "png":
        return ".png"
    elif format == "webp":
        return ".webp"
    elif format == "jpeg":
        return ".jpg"
    else:
        raise NotImplementedError(format)


def save_image(im, output_filename, format="png", png_info=None):
    if format == "png":
        options = {
            "compress_level": 6,
            "pnginfo": png_info,
        }
    elif format == "webp":
        options = {
            "quality": 95,
            "method": 4,
            "lossless": True
        }
    elif format == "jpeg":
        options = {
            "quality": 95,
            "subsampling": "4:2:0",
        }
    else:
        raise NotImplementedError(format)

    im.save(output_filename, format=format, **options)


def preprocess_image(x, args):
    if args.rotate_left:
        x = torch.rot90(x, 1, (-2, -1))
    elif args.rotate_right:
        x = torch.rot90(x, 3, (-2, -1))

    h, w = x.shape[-2:]
    new_w, new_h = w, h
    if args.max_output_height is not None and new_h > args.max_output_height:
        new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height
        # only apply max height
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        if x.ndim == 3:
            x = F.interpolate(x.unsqueeze(0), (new_h, new_w),
                              mode="bicubic", antialias=True, align_corners=True).squeeze(0)
        elif x.ndim == 4:
            x = F.interpolate(x, (new_h, new_w),
                              mode="bicubic", antialias=True, align_corners=True)

        x = torch.clamp(x, 0, 1)

    return x


def apply_divergence(depth, im, args, side_model):
    batch = True
    if depth.ndim != 4:
        # CHW
        depth = depth.unsqueeze(0)
        im = im.unsqueeze(0)
        batch = False
    else:
        # BCHW
        pass

    if args.method in {"grid_sample", "backward"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_grid_sample(
            im, depth,
            args.divergence, convergence=args.convergence,
            synthetic_view=args.synthetic_view)
    elif args.method in {"forward", "forward_fill"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_forward_warp(
            im, depth,
            args.divergence, convergence=args.convergence,
            method=args.method, synthetic_view=args.synthetic_view,
            inpaint_model=args.state["inpaint_model"])
    else:
        if args.stereo_width is not None:
            # NOTE: use src aspect ratio instead of depth aspect ratio
            H, W = im.shape[2:]
            stereo_width = min(W, args.stereo_width)
            if depth.shape[3] != stereo_width:
                new_w = stereo_width
                new_h = int(H * (stereo_width / W))
                depth = F.interpolate(depth, size=(new_h, new_w),
                                      mode="bilinear", align_corners=True, antialias=True)
                depth = torch.clamp(depth, 0, 1)
        left_eye, right_eye = apply_divergence_nn_LR(
            side_model, im, depth,
            args.divergence, args.convergence, args.warp_steps,
            mapper=args.mapper,
            synthetic_view=args.synthetic_view,
            preserve_screen_border=args.preserve_screen_border,
            enable_amp=not args.disable_amp)

    if not batch:
        left_eye = left_eye.squeeze(0)
        right_eye = right_eye.squeeze(0)

    return left_eye, right_eye


def postprocess_padding(left_eye, right_eye, pad, pad_mode):
    assert pad_mode in {"tblr", "tb", "lr", "16:9"}
    if pad_mode in {"tblr", "tb", "lr"}:
        pad_h = pad_w = 0
        if "tb" in pad_mode:
            pad_h = round(left_eye.shape[1] * pad) // 2
        if "lr" in pad_mode:
            pad_w = round(left_eye.shape[2] * pad) // 2
        left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
    elif pad_mode == "16:9":
        # fit to 16:9
        # pad size is ignored
        eps = 1e-3
        target_ratio = 16 / 9
        height, width = left_eye.shape[1:]
        current_ratio = width / height
        if abs(target_ratio - current_ratio) > eps:
            pad_h = pad_w = 0
            if current_ratio > target_ratio:
                # pad top-bottom
                target_height = round(width / target_ratio)
                pad_h = (target_height - height) // 2
            else:
                # pad left-right
                target_width = round(height * target_ratio)
                pad_w = (target_width - width) // 2
            left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
            right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
    return left_eye, right_eye


def postprocess_image(left_eye, right_eye, args):
    # CHW
    ipd_pad = int(abs(args.ipd_offset) * 0.01 * left_eye.shape[2])
    ipd_pad -= ipd_pad % 2
    if ipd_pad > 0 and not (args.rgbd or args.half_rgbd):
        pad_o, pad_i = (ipd_pad * 2, ipd_pad) if args.ipd_offset > 0 else (ipd_pad, ipd_pad * 2)
        left_eye = TF.pad(left_eye, (pad_o, 0, pad_i, 0), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_i, 0, pad_o, 0), padding_mode="constant")

    if args.pad is not None or args.pad_mode == "16:9":
        left_eye, right_eye = postprocess_padding(left_eye, right_eye, pad=args.pad, pad_mode=args.pad_mode)
    if args.vr180:
        left_eye = equirectangular_projection(left_eye, device=left_eye.device)
        right_eye = equirectangular_projection(right_eye, device=right_eye.device)
    elif args.half_sbs or args.half_rgbd:
        left_eye = TF.resize(left_eye, (left_eye.shape[1], left_eye.shape[2] // 2),
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
        right_eye = TF.resize(right_eye, (right_eye.shape[1], right_eye.shape[2] // 2),
                              interpolation=InterpolationMode.BICUBIC, antialias=True)
    elif args.half_tb:
        left_eye = TF.resize(left_eye, (left_eye.shape[1] // 2, left_eye.shape[2]),
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
        right_eye = TF.resize(right_eye, (right_eye.shape[1] // 2, right_eye.shape[2]),
                              interpolation=InterpolationMode.BICUBIC, antialias=True)

    if args.anaglyph is not None:
        # Anaglyph
        sbs = apply_anaglyph_redcyan(left_eye, right_eye, args.anaglyph)
    elif args.tb or args.half_tb:
        # TopBottom
        sbs = torch.cat([left_eye, right_eye], dim=1)
        sbs = torch.clamp(sbs, 0., 1.)
    elif args.cross_eyed:
        # Reverse SideBySide
        sbs = torch.cat([right_eye, left_eye], dim=2)
        sbs = torch.clamp(sbs, 0., 1.)
    else:
        # SideBySide or RGBD
        sbs = torch.cat([left_eye, right_eye], dim=2)
        sbs = torch.clamp(sbs, 0., 1.)

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


def debug_depth_image(depth, args):
    depth = depth.float()
    mean_depth, std_depth = depth.mean().item(), depth.std().item()
    depth2 = get_mapper(args.mapper)(depth)
    out = torch.cat([depth, depth2], dim=2).cpu()
    out = out.repeat((3, 1, 1))
    # gc = ImageDraw.Draw(out)
    # gc.text((16, 16), (f"min={round(float(depth_min), 4)}\n"
    #                    f"max={round(float(depth_max), 4)}\n"
    #                    f"mean={round(float(mean_depth), 4)}\n"
    #                    f"std={round(float(std_depth), 4)}"), "gray")

    return out


def process_image(x, args, depth_model, side_model):
    assert depth_model.get_ema_buffer_size() == 1
    with torch.inference_mode():
        x = preprocess_image(x, args)
        depth = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                  enable_amp=not args.disable_amp,
                                  edge_dilation=args.edge_dilation,
                                  depth_aa=args.depth_aa)
        depth = depth_model.minmax_normalize_chw(depth)

        if args.debug_depth:
            return debug_depth_image(depth, args)
        elif args.rgbd or args.half_rgbd:
            left_eye, right_eye = apply_rgbd(x, depth, mapper=args.mapper)
            sbs = postprocess_image(left_eye, right_eye, args)
            return sbs
        else:
            left_eye, right_eye = apply_divergence(depth, x, args, side_model)
            sbs = postprocess_image(left_eye, right_eye, args)
            return sbs


def process_images(files, output_dir, args, depth_model, side_model, title=None):
    # disable ema minmax for each process
    depth_model.disable_ema()
    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        # skip existing output files
        remaining_files = []
        existing_files = []
        for fn in files:
            output_filename = path.join(
                output_dir,
                make_output_filename(path.basename(fn), args, video=False))
            if not path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb", "exif_transpose": not args.disable_exif_transpose})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title)
    stop_event = args.state["stop_event"]
    suspend_event = args.state["suspend_event"]

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool:
        for im, meta in loader:
            filename = meta["filename"]
            output_filename = path.join(
                output_dir,
                make_output_filename(filename, args, video=False))
            if im is None:
                pbar.update(1)
                continue
            im = TF.to_tensor(im).to(args.state["device"])
            output = process_image(im, args, depth_model, side_model)
            output = to_pil_image(output)
            f = pool.submit(save_image, output, output_filename, format=args.format)
            #  f.result() # for debug
            futures.append(f)
            pbar.update(1)
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []
        for f in futures:
            f.result()
    pbar.close()


# video callbacks

def bind_single_frame_callback(depth_model, side_model, segment_pts, args):
    src_queue = []
    frame_cpu_offload = depth_model.get_ema_buffer_size() > 1
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    def _postprocess(depths):
        frames = []
        for depth in depths:
            x, pts = src_queue.pop(0)
            if frame_cpu_offload:
                x = x.to(args.state["device"]).permute(2, 0, 1) / torch.iinfo(x.dtype).max
            if args.debug_depth:
                out = debug_depth_image(depth, args)
            elif args.rgbd or args.half_rgbd:
                left_eye, right_eye = apply_rgbd(x, depth, mapper=args.mapper)
                out = postprocess_image(left_eye, right_eye, args)
            else:
                left_eye, right_eye = apply_divergence(depth, x, args, side_model)
                out = postprocess_image(left_eye, right_eye, args)

            if pts in segment_pts:
                if args.debug_depth:
                    # debug red line
                    out[0, 0:8, :] = 1.0
            frames.append(VU.to_frame(out, use_16bit=use_16bit))
        return frames

    @torch.inference_mode()
    def _frame_callback(frame):
        if frame is None:
            # flush
            return _postprocess(depth_model.flush_minmax_normalize())

        frame_hwc = torch.from_numpy(VU.to_ndarray(frame))
        pix_dtype, pix_max = frame_hwc.dtype, torch.iinfo(frame_hwc.dtype).max

        if frame_cpu_offload:
            # cpu buffer
            if args.max_output_height is not None or args.rotate_right or args.rotate_left:
                x = frame_hwc.to(args.state["device"]).permute(2, 0, 1) / pix_max
                x = preprocess_image(x, args)
                frame_hwc = (x.permute(1, 2, 0) * pix_max).round().clamp(0, pix_max).to(pix_dtype).cpu()
                src_queue.append((frame_hwc, frame.pts))
            else:
                src_queue.append((frame_hwc, frame.pts))
                x = frame_hwc.to(args.state["device"]).permute(2, 0, 1) / pix_max
        else:
            # gpu buffer
            x = frame_hwc.to(args.state["device"]).permute(2, 0, 1) / pix_max
            x = preprocess_image(x, args)
            src_queue.append((x, frame.pts))

        depth = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                  enable_amp=not args.disable_amp,
                                  edge_dilation=args.edge_dilation,
                                  depth_aa=args.depth_aa)
        depth = depth_model.minmax_normalize_chw(depth)
        depths = [depth] if depth is not None else []
        if frame.pts in segment_pts:
            depths += depth_model.flush_minmax_normalize()
            depth_model.reset_state()

        return _postprocess(depths)

    return _frame_callback


def bind_batch_frame_callback(depth_model, side_model, segment_pts, args):
    depth_lock = threading.RLock()
    sbs_lock = threading.RLock()
    enqueue_ticket_lock = TicketLock()
    dequeue_ticket_lock = TicketLock()
    streams = threading.local()
    src_queue = []
    frame_cpu_offload = depth_model.get_ema_buffer_size() > 1
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    def _postprocess(depth_batch, reset_ema, dequeue_ticket_id, flush, device):
        src_depth_pairs = []
        # Reorder threads
        with dequeue_ticket_lock(dequeue_ticket_id):
            with depth_lock:
                if flush:
                    depth_list = depth_model.flush_minmax_normalize()
                else:
                    depth_list = depth_model.minmax_normalize(depth_batch, reset_ema=reset_ema)

            for depths in chunks(depth_list, args.batch_size):
                if isinstance(depths, list):
                    depths = torch.stack([depth.to(device) for depth in depths])
                else:
                    depths = depths.to(device)
                if frame_cpu_offload:
                    x_srcs = torch.stack([src_queue.pop(0)[0] for _ in range(len(depths))])
                else:
                    x_srcs, _ = src_queue.pop(0)

                src_depth_pairs.append((x_srcs, depths))

        results = []
        for x_srcs, depths in src_depth_pairs:
            if frame_cpu_offload:
                x_srcs = x_srcs.to(device).permute(0, 3, 1, 2) / torch.iinfo(x_srcs.dtype).max

            with sbs_lock:  # TODO: unclear whether this is actually needed
                if args.rgbd or args.half_rgbd:
                    left_eyes, right_eyes = apply_rgbd(x_srcs, depths, mapper=args.mapper)
                else:
                    if args.method in {"forward_fill", "forward"}:
                        # lock all threads (sbs_lock -> ticket_lock -> depth_lock order)
                        with enqueue_ticket_lock, dequeue_ticket_lock, depth_lock:
                            left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model)
                    else:
                        left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model)

            frames = [postprocess_image(left_eyes[i], right_eyes[i], args)
                      for i in range(left_eyes.shape[0])]
            results += [VU.to_frame(frame, use_16bit=use_16bit) for frame in frames]

        return results

    def _batch_infer(x, pts, flush, enqueue_ticket_id):
        # Reorder threads
        with enqueue_ticket_lock(enqueue_ticket_id):
            dequeue_ticket_id = dequeue_ticket_lock.new_ticket()
            if not flush:
                x = preprocess_image(x, args)
                if frame_cpu_offload:
                    if use_16bit:
                        x_cpu = (x.permute(0, 2, 3, 1) * 65535.0).round().clamp(0, 65535).to(torch.uint16).cpu()
                    else:
                        x_cpu = (x.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
                    for x_, pts_ in zip(x_cpu, pts):
                        src_queue.append((x_, pts_))
                else:
                    src_queue.append((x, pts))

        if flush:
            return None, dequeue_ticket_id
        else:
            with depth_lock:
                depth_batch = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                                enable_amp=not args.disable_amp,
                                                edge_dilation=args.edge_dilation,
                                                depth_aa=args.depth_aa)
            return depth_batch, dequeue_ticket_id

    @torch.inference_mode()
    def _cuda_stream_wrapper(preprocess_args):
        x, pts, flush, enqueue_ticket_id = preprocess_args
        if flush:
            device = args.state["device"]
            reset_ema = None
            depth_batch, dequeue_ticket_id = _batch_infer(
                None, None, flush=flush, enqueue_ticket_id=enqueue_ticket_id)
        else:
            device = x.device
            reset_ema = [t in segment_pts for t in pts]
            if args.cuda_stream and device_is_cuda(x.device):
                device_name = str(device)
                if not hasattr(streams, device_name):
                    setattr(streams, device_name, torch.cuda.Stream(device=x.device))
                stream = getattr(streams, device_name)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.device(x.device), torch.cuda.stream(stream):
                    depth_batch, dequeue_ticket_id = _batch_infer(
                        x, pts, flush=flush, enqueue_ticket_id=enqueue_ticket_id)
                    stream.synchronize()
            else:
                depth_batch, dequeue_ticket_id = _batch_infer(
                    x, pts, flush=flush, enqueue_ticket_id=enqueue_ticket_id)

        return _postprocess(
            depth_batch, reset_ema,
            dequeue_ticket_id=dequeue_ticket_id,
            flush=flush,
            device=device
        )

    def _preprocess(x, pts, flush):
        enqueue_ticket_id = enqueue_ticket_lock.new_ticket()
        return (x, pts, flush, enqueue_ticket_id)

    return _cuda_stream_wrapper, _preprocess


def bind_vda_frame_callback(depth_model, side_model, segment_pts, args):
    src_queue = []
    batch_queue = []
    pts_queue = []
    depth_model.reset()
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    def _postprocess(depth_list):
        results = []
        if args.debug_depth:
            for depth in depth_list:
                out = debug_depth_image(depth, args)
                _, pts = src_queue.pop(0)
                if pts in segment_pts:
                    out[0, 0:8, :] = 1.0
                results.append(VU.to_frame(out, use_16bit=use_16bit))
        else:
            for depths in chunks(depth_list, args.batch_size):
                depths = torch.stack(depths)
                x_srcs = [src_queue.pop(0)[0] for _ in range(len(depths))]
                x_srcs = torch.stack(x_srcs).to(args.state["device"]).permute(0, 3, 1, 2)
                x_srcs = x_srcs / torch.iinfo(x_srcs.dtype).max
                if args.rgbd or args.half_rgbd:
                    left_eyes, right_eyes = apply_rgbd(x_srcs, depths, mapper=args.mapper)
                else:
                    left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model)
                frames = [postprocess_image(left_eyes[i], right_eyes[i], args)
                          for i in range(left_eyes.shape[0])]
                results += [VU.to_frame(frame, use_16bit=use_16bit) for frame in frames]
        return results

    def _batch_infer():
        x = torch.stack(batch_queue).to(args.state["device"]).permute(0, 3, 1, 2)
        pix_dtype, pix_max = x.dtype, torch.iinfo(x.dtype).max
        x = x / pix_max
        if args.max_output_height is not None or args.rotate_right or args.rotate_left:
            x = preprocess_image(x, args)
            x_srcs = (x.permute(0, 2, 3, 1) * pix_max).round().clamp(0, pix_max).to(pix_dtype).cpu()
        else:
            x_srcs = batch_queue

        for i, x_src in enumerate(x_srcs):
            src_queue.append((x_src, pts_queue[i]))

        depth_list = depth_model.infer_with_normalize(
            x, pts_queue, segment_pts,
            enable_amp=not args.disable_amp,
            edge_dilation=args.edge_dilation,
            depth_aa=args.depth_aa)

        pts_queue.clear()
        batch_queue.clear()

        return _postprocess(depth_list)

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            # flush
            results = []
            if batch_queue:
                results += _batch_infer()
            depth_list = depth_model.flush_with_normalize(
                enable_amp=not args.disable_amp,
                edge_dilation=args.edge_dilation,
                depth_aa=args.depth_aa)
            results += _postprocess(depth_list)
            return [VU.to_frame(new_frame, use_16bit=use_16bit) for new_frame in results]

        frame_hwc = torch.from_numpy(VU.to_ndarray(frame))
        batch_queue.append(frame_hwc)
        pts_queue.append(frame.pts)

        if len(batch_queue) == args.batch_size:
            results = _batch_infer()
            return [VU.to_frame(new_frame, use_16bit=use_16bit) for new_frame in results]
        else:
            return None

    return frame_callback


def process_video_full(input_filename, output_path, args, depth_model, side_model):
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)
    is_video_depth_anything = depth_model.get_name() == "VideoDepthAnything"
    is_video_depth_anything_streaming = depth_model.get_name() == "VideoDepthAnythingStreaming"
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)

    if args.compile and side_model is not None and not isinstance(side_model, DeviceSwitchInference):
        side_model = compile_model(side_model)

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    make_parent_dir(output_filename)
    if args.scene_detect:
        with TorchHubDir(HUB_MODEL_DIR):
            segment_pts = SBD.detect_boundary(
                input_filename,
                max_fps=args.max_fps,
                device=args.state["device"],
                start_time=args.start_time,
                end_time=args.end_time,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                tqdm_title=f"{path.basename(input_filename)}: Scene Boundary Detection",
            )
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                return
        gc_collect()
    else:
        segment_pts = set()

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            options=make_video_codec_option(args),
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        )

    @torch.inference_mode()
    def test_callback(frame):
        decay, buffer_size = depth_model.get_ema_state()
        depth_model.disable_ema()
        x = VU.to_tensor(frame, device=args.state["device"])
        x = process_image(x, args, depth_model, side_model)
        if ema_normalize:
            # reset ema to avoid affecting test frames
            depth_model.enable_ema(decay=decay, buffer_size=buffer_size)
        depth_model.reset()
        return VU.to_frame(x, use_16bit=use_16bit)

    if is_video_depth_anything:
        with depth_model.compile_context(enabled=args.compile):
            VU.process_video(input_filename, output_filename,
                             config_callback=config_callback,
                             frame_callback=bind_vda_frame_callback(
                                 depth_model=depth_model,
                                 side_model=side_model,
                                 segment_pts=segment_pts,
                                 args=args
                             ),
                             test_callback=test_callback,
                             vf=args.vf,
                             stop_event=args.state["stop_event"],
                             suspend_event=args.state["suspend_event"],
                             tqdm_fn=args.state["tqdm_fn"],
                             title=path.basename(input_filename),
                             start_time=args.start_time,
                             end_time=args.end_time)

    elif args.low_vram or args.debug_depth or is_video_depth_anything_streaming:
        with depth_model.compile_context(enabled=args.compile):
            VU.process_video(input_filename, output_filename,
                             config_callback=config_callback,
                             frame_callback=bind_single_frame_callback(
                                 depth_model=depth_model,
                                 side_model=side_model,
                                 segment_pts=segment_pts,
                                 args=args
                             ),
                             test_callback=test_callback,
                             vf=args.vf,
                             stop_event=args.state["stop_event"],
                             suspend_event=args.state["suspend_event"],
                             tqdm_fn=args.state["tqdm_fn"],
                             title=path.basename(input_filename),
                             start_time=args.start_time,
                             end_time=args.end_time)
    else:
        extra_queue = 1 if len(args.state["devices"]) == 1 else 0
        minibatch_size = args.batch_size // 2 or 1 if args.tta else args.batch_size

        frame_callback, preprocess_callback = bind_batch_frame_callback(
            depth_model=depth_model,
            side_model=side_model,
            segment_pts=segment_pts,
            args=args
        )
        frame_callback = VU.FrameCallbackPool(
            frame_callback=frame_callback,
            preprocess_callback=preprocess_callback,
            batch_size=minibatch_size,
            device=args.state["devices"],
            max_workers=args.max_workers,
            max_batch_queue=args.max_workers + extra_queue,
            require_pts=True,
            require_flush=True,
            use_16bit=use_16bit,
        )
        try:
            with depth_model.compile_context(enabled=args.compile):
                VU.process_video(input_filename, output_filename,
                                 config_callback=config_callback,
                                 frame_callback=frame_callback,
                                 test_callback=test_callback,
                                 vf=args.vf,
                                 stop_event=args.state["stop_event"],
                                 suspend_event=args.state["suspend_event"],
                                 tqdm_fn=args.state["tqdm_fn"],
                                 title=path.basename(input_filename),
                                 start_time=args.start_time,
                                 end_time=args.end_time)
        finally:
            frame_callback.shutdown()


def process_video_keyframes(input_filename, output_path, args, depth_model, side_model):
    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    output_dir = path.join(path.dirname(output_filename), path.splitext(path.basename(output_filename))[0])
    if output_dir.endswith("_LRF"):
        output_dir = output_dir[:-4]
    os.makedirs(output_dir, exist_ok=True)

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool:
        futures = []

        def frame_callback(frame):
            im = TF.to_tensor(frame.to_image()).to(args.state["device"])
            output = process_image(im, args, depth_model, side_model)
            output = to_pil_image(output)
            output_filename = path.join(
                output_dir,
                path.basename(output_dir) + "_" + str(frame.pts).zfill(8) + FULL_SBS_SUFFIX + get_image_ext(args.format))
            f = pool.submit(save_image, output, output_filename, format=args.format)
            futures.append(f)
        VU.process_video_keyframes(input_filename, frame_callback=frame_callback,
                                   min_interval_sec=args.keyframe_interval,
                                   stop_event=args.state["stop_event"],
                                   suspend_event=args.state["suspend_event"],
                                   title=path.basename(input_filename))
        for f in futures:
            f.result()


def process_video(input_filename, output_path, args, depth_model, side_model):
    # disable ema minmax for each process
    depth_model.reset()
    depth_model.disable_ema()

    if args.keyframe:
        process_video_keyframes(input_filename, output_path, args, depth_model, side_model)
    else:
        process_video_full(input_filename, output_path, args, depth_model, side_model)


def export_images(input_path, output_dir, args, title=None):
    if path.isdir(input_path):
        files = ImageLoader.listdir(input_path)
        src_rgb_dir = path.normpath(path.abspath(input_path))
    else:
        assert is_image(input_path)
        files = [input_path]
        src_rgb_dir = path.normpath(path.abspath(path.dirname(input_path)))

    if not files:
        # no image files
        return

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
    if args.export_depth_only:
        config.rgb_dir = src_rgb_dir
        rgb_dir = src_rgb_dir
    else:
        rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    config_file = path.join(output_dir, export_config.FILENAME)

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    depth_model = args.state["depth_model"]
    depth_model.disable_ema()

    if args.resume:
        # skip existing depth files
        remaining_files = []
        existing_files = []
        for fn in files:
            basename = path.splitext(path.basename(fn))[0] + ".png"
            depth_file = path.join(depth_dir, basename)
            if not path.exists(depth_file):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)

        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb", "exif_transpose": not args.disable_exif_transpose})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title or "Images")
    stop_event = args.state["stop_event"]
    suspend_event = args.state["suspend_event"]

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool, torch.inference_mode():
        for im, meta in loader:
            basename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            rgb_file = path.join(rgb_dir, basename)
            depth_file = path.join(depth_dir, basename)
            if im is None:
                pbar.update(1)
                continue
            im = TF.to_tensor(im).to(args.state["device"])
            im = preprocess_image(im, args)
            depth = depth_model.infer(im, tta=args.tta, low_vram=args.low_vram,
                                      enable_amp=not args.disable_amp,
                                      edge_dilation=edge_dilation,
                                      depth_aa=args.depth_aa)
            depth = depth_model.minmax_normalize_chw(depth)
            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = F.interpolate(depth.unsqueeze(0), size=(im.shape[1], im.shape[2]),
                                      mode="bilinear", antialias=True, align_corners=True).squeeze(0)
            futures.append(pool.submit(depth_model.save_normalized_depth, depth, depth_file))

            if not args.export_depth_only:
                futures.append(pool.submit(save_image, to_pil_image(im), rgb_file))

            pbar.update(1)
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []

        for f in futures:
            f.result()
    pbar.close()
    config.save(config_file)


def get_resume_seq(depth_dir, rgb_dir):
    last_seq = -1
    depth_files = sorted([path.basename(fn) for fn in ImageLoader.listdir(depth_dir)])
    if rgb_dir:
        rgb_files = sorted([path.basename(fn) for fn in ImageLoader.listdir(rgb_dir)])
        if rgb_files and depth_files:
            last_seq = int(path.splitext(min(rgb_files[-1], depth_files[-1]))[0], 10)
    else:
        if depth_files:
            last_seq = int(path.splitext(depth_files[-1])[0], 10)

    return last_seq


# export callbacks


def bind_export_single_frame_callback(depth_model, segment_pts, rgb_dir, depth_dir, pool, args):
    src_queue = []
    futures = []
    if args.export_disparity:
        edge_dilation = args.edge_dilation
    else:
        edge_dilation = 0

    def _postprocess(depths):
        for depth in depths:
            x, pts = src_queue.pop(0)

            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = TF.resize(depth, size=(x.shape[0], x.shape[1]),
                                  interpolation=InterpolationMode.BILINEAR, antialias=True)
            seq = str(pts).zfill(8)
            futures.append(
                pool.submit(depth_model.save_normalized_depth, depth, path.join(depth_dir, f"{seq}.png"))
            )
            if not args.export_depth_only:
                im = Image.fromarray(x)
                futures.append(
                    pool.submit(save_image, im, path.join(rgb_dir, f"{seq}.png"))
                )

            if len(futures) >= IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures.clear()

        return None

    @torch.inference_mode()
    def _frame_callback(frame):
        if frame is None:
            # flush
            return _postprocess(depth_model.flush_minmax_normalize())

        frame_np = frame.to_ndarray(format="rgb24")
        frame_hwc8 = torch.from_numpy(frame_np)
        src_queue.append((frame_np, frame.pts))
        x = frame_hwc8.to(args.state["device"]).permute(2, 0, 1) / 255.0
        x = preprocess_image(x, args)
        depth = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                  enable_amp=not args.disable_amp,
                                  edge_dilation=edge_dilation,
                                  depth_aa=args.depth_aa)
        depth = depth_model.minmax_normalize_chw(depth)
        depths = [depth] if depth is not None else []
        if frame.pts in segment_pts:
            depths += depth_model.flush_minmax_normalize()
            depth_model.reset_state()

        return _postprocess(depths)

    return _frame_callback


def bind_export_vda_frame_callback(depth_model, segment_pts, rgb_dir, depth_dir, pool, args):
    src_queue = []
    batch_queue = []
    pts_queue = []
    futures = []

    depth_model.reset()
    if args.export_disparity:
        edge_dilation = args.edge_dilation
    else:
        edge_dilation = 0

    def _postprocess(depth_list):
        for depth in depth_list:
            x, pts = src_queue.pop(0)

            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = TF.resize(depth, size=(x.shape[0], x.shape[1]),
                                  interpolation=InterpolationMode.BILINEAR, antialias=True)
            seq = str(pts).zfill(8)
            futures.append(
                pool.submit(depth_model.save_normalized_depth, depth, path.join(depth_dir, f"{seq}.png"))
            )
            if not args.export_depth_only:
                im = Image.fromarray(x.numpy())
                futures.append(
                    pool.submit(save_image, im, path.join(rgb_dir, f"{seq}.png"))
                )

            if len(futures) >= IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures.clear()

        return None

    def _batch_infer():
        x = torch.stack(batch_queue).to(args.state["device"]).permute(0, 3, 1, 2) / 255.0
        if args.max_output_height is not None or args.rotate_right or args.rotate_left:
            x = preprocess_image(x, args)
            x_srcs = (x.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        else:
            x_srcs = batch_queue

        for i, x_src in enumerate(x_srcs):
            src_queue.append((x_src, pts_queue[i]))

        depth_list = depth_model.infer_with_normalize(
            x, pts_queue, segment_pts,
            enable_amp=not args.disable_amp,
            edge_dilation=edge_dilation,
            depth_aa=args.depth_aa)

        pts_queue.clear()
        batch_queue.clear()

        return _postprocess(depth_list)

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            # flush
            if batch_queue:
                _batch_infer()
            depth_list = depth_model.flush_with_normalize(
                enable_amp=not args.disable_amp,
                edge_dilation=edge_dilation,
                depth_aa=args.depth_aa)
            _postprocess(depth_list)
        else:
            frame_hwc8 = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            batch_queue.append(frame_hwc8)
            pts_queue.append(frame.pts)

            if len(batch_queue) == args.batch_size:
                _batch_infer()

    return frame_callback


def export_video(input_filename, output_dir, args, title=None):
    basename = path.splitext(path.basename(input_filename))[0]
    title = title or path.basename(input_filename)
    if args.export_disparity:
        mapper = "none"
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
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
                "export_depth_only": args.export_depth_only,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "depth_aa": args.depth_aa,
                "max_fps": args.max_fps,
                "ema_normalize": args.ema_normalize,
                "ema_decay": args.ema_decay,
                "ema_buffer": args.ema_buffer,
                "scene_detect": args.scene_detect,
            }
        }
    )
    # NOTE: Windows does not allow creating folders with trailing spaces. basename.strip()
    output_dir = path.join(output_dir, basename.strip())
    rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    audio_file = path.join(output_dir, config.audio_file)
    config_file = path.join(output_dir, export_config.FILENAME)

    if not args.resume and (not args.yes and path.exists(config_file)):
        y = input(f"File '{config_file}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    if args.export_depth_only:
        rgb_dir = None
        config.rgb_dir = None
    else:
        os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    if args.scene_detect:
        with TorchHubDir(HUB_MODEL_DIR):
            segment_pts = SBD.detect_boundary(
                input_filename,
                max_fps=args.max_fps,
                device=args.state["device"],
                start_time=args.start_time,
                end_time=args.end_time,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                tqdm_title=f"{path.basename(input_filename)}: Scene Boundary Detection",
            )
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                return
        gc_collect()
    else:
        segment_pts = set()
    config.user_data["scene_boundary"] = ",".join([str(pts).zfill(8) for pts in sorted(list(segment_pts))])

    if args.resume:
        resume_seq = get_resume_seq(depth_dir, rgb_dir) - args.batch_size
    else:
        resume_seq = -1

    if resume_seq > 0 and path.exists(audio_file):
        has_audio = True
    else:
        if args.export_depth_only:
            has_audio = False
        else:
            has_audio = VU.export_audio(input_filename, audio_file,
                                        start_time=args.start_time, end_time=args.end_time,
                                        title=f"{title} Audio",
                                        stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
                                        tqdm_fn=args.state["tqdm_fn"])
    if not has_audio:
        config.audio_file = None

    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
        return

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        config.fps = fps  # update fps

        def state_update_callback(c):
            config.source_color_range = c.state["source_color_range"]
            config.output_colorspace = c.state["output_colorspace"]

        video_output_config = VU.VideoOutputConfig(fps=fps, pix_fmt=args.pix_fmt, colorspace=args.colorspace)
        video_output_config.state_updated = state_update_callback

        return video_output_config

    depth_model = args.state["depth_model"]
    depth_model.reset()
    depth_model.disable_ema()
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)

    with depth_model.compile_context(enabled=args.compile):
        max_workers = max(args.max_workers, 8)

        with PoolExecutor(max_workers=max_workers) as pool:
            if args.state["depth_model"].get_name() == "VideoDepthAnything":
                frame_callback = bind_export_vda_frame_callback(
                    depth_model=depth_model,
                    segment_pts=segment_pts,
                    rgb_dir=rgb_dir,
                    depth_dir=depth_dir,
                    pool=pool,
                    args=args,
                )
            else:
                frame_callback = bind_export_single_frame_callback(
                    depth_model=depth_model,
                    segment_pts=segment_pts,
                    rgb_dir=rgb_dir,
                    depth_dir=depth_dir,
                    pool=pool,
                    args=args,
                )

            VU.hook_frame(input_filename,
                          config_callback=config_callback,
                          frame_callback=frame_callback,
                          vf=args.vf,
                          stop_event=args.state["stop_event"],
                          suspend_event=args.state["suspend_event"],
                          tqdm_fn=args.state["tqdm_fn"],
                          title=title,
                          start_time=args.start_time,
                          end_time=args.end_time)
        config.save(config_file)


def process_config_video(config, args, side_model):
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, audio_file = config.resolve_paths(base_dir)

    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        basename = config.basename or path.basename(base_dir)
        output_filename = path.join(
            args.output,
            make_output_filename(basename, args, video=True))
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
        load_func=BaseDepthModel.load_depth)
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
        depth = BaseDepthModel.load_depth(depth_file)[0].to(args.state["device"])
        rgb = TF.to_tensor(rgb).to(args.state["device"])
        frame = batch_callback(rgb.unsqueeze(0), depth.unsqueeze(0))
        return frame.shape[2:]

    minibatch_size = args.batch_size // 2 or 1 if args.tta else args.batch_size

    def generator():
        rgb_batch = []
        depth_batch = []
        for rgb, depth in zip(rgb_loader, depth_loader):
            rgb = TF.to_tensor(rgb[0])
            rgb_batch.append(rgb)
            depth_batch.append(depth[0])
            if len(rgb_batch) == minibatch_size:
                frames = batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                        torch.stack(depth_batch).to(args.state["device"]))
                rgb_batch.clear()
                depth_batch.clear()

                yield [VU.to_frame(frame, use_16bit=use_16bit) for frame in frames]

        if rgb_batch:
            frames = batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                    torch.stack(depth_batch).to(args.state["device"]))
            rgb_batch.clear()
            depth_batch.clear()

            yield [VU.to_frame(frame, use_16bit=use_16bit) for frame in frames]

    output_height, output_width = test_output_size(rgb_files[0], depth_files[0])
    video_config = VU.VideoOutputConfig(
        fps=config.fps,  # use config.fps, ignore args.max_fps
        container_format=args.video_format,
        video_codec=args.video_codec,
        pix_fmt=args.pix_fmt,
        colorspace=args.colorspace,
        options=make_video_codec_option(args),
        container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        output_width=output_width,
        output_height=output_height
    )
    video_config.state["source_color_range"] = config.source_color_range
    video_config.state["output_colorspace"] = config.output_colorspace

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
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
        )
    finally:
        args.mapper = original_mapper


def process_config_images(config, args, side_model):
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, _ = config.resolve_paths(base_dir)

    def fix_rgb_depth_pair(rgb_files, depth_files):
        rgb_db = {path.splitext(path.basename(fn))[0]: fn for fn in rgb_files}
        depth_db = {path.splitext(path.basename(fn))[0]: fn for fn in depth_files}
        and_keys = sorted(list(rgb_db.keys() & depth_db.keys()))
        rgb_files = [rgb_db[key] for key in and_keys if key in rgb_db]
        depth_files = [depth_db[key] for key in and_keys if key in depth_db]
        return rgb_files, depth_files

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)

    if args.resume:
        # skip existing output files
        remaining_files = []
        existing_files = []
        for fn in rgb_files:
            output_filename = path.join(
                output_dir,
                make_output_filename(path.basename(fn), args, video=False))
            if not path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        rgb_files = remaining_files

    rgb_files, depth_files = fix_rgb_depth_pair(rgb_files, depth_files)

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
        load_func=BaseDepthModel.load_depth)

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
            stop_event = args.state["stop_event"]
            suspend_event = args.state["suspend_event"]
            futures = []
            for (rgb, rgb_meta), (depth, depth_meta) in zip(rgb_loader, depth_loader):
                rgb_filename = path.splitext(path.basename(rgb_meta["filename"]))[0]
                depth_filename = path.splitext(path.basename(depth_meta["filename"]))[0]
                if rgb_filename != depth_filename:
                    raise ValueError(f"No match {rgb_filename} and {depth_filename}")
                rgb = TF.to_tensor(rgb).to(args.state["device"])
                depth = depth.to(args.state["device"])
                if not config.skip_edge_dilation and args.edge_dilation > 0:
                    depth = -dilate_edge(-depth.unsqueeze(0), args.edge_dilation).squeeze(0)

                left_eye, right_eye = apply_divergence(depth, rgb, args, side_model)
                sbs = postprocess_image(left_eye, right_eye, args)
                sbs = to_pil_image(sbs)

                output_filename = path.join(
                    output_dir,
                    make_output_filename(rgb_filename, args, video=False))
                f = pool.submit(save_image, sbs, output_filename, format=args.format)
                futures.append(f)
                pbar.update(1)
                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
                if len(futures) > IMAGE_IO_QUEUE_MAX:
                    for f in futures:
                        f.result()
                    futures = []
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
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser.add_argument("--input", "-i", type=str, required=required_true,
                        help="input file or directory")
    parser.add_argument("--output", "-o", type=str, required=required_true,
                        help="output file or directory")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--compile", action="store_true", help="compile model if possible")
    parser.add_argument("--method", type=str, default="row_flow",
                        choices=["grid_sample", "backward", "forward", "forward_fill",
                                 "mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s",
                                 "row_flow", "row_flow_sym",
                                 "row_flow_v3", "row_flow_v3_sym",
                                 "row_flow_v2"],
                        help="left-right divergence method")
    parser.add_argument("--inpaint-model", type=str, default=None,
                        help="path for inpaint model")
    parser.add_argument("--synthetic-view", type=str, default="both", choices=["both", "right", "left"],
                        help=("the side that generates synthetic view."
                              "when `right`, the left view will be the original input image/frame"
                              " and only the right will be synthesized."))
    parser.add_argument("--preserve-screen-border", action="store_true",
                        help=("force set screen border parallax to zero"))
    parser.add_argument("--divergence", "-d", type=float, default=2.0,
                        help=("strength of 3D effect. 0-2 is reasonable value"))
    parser.add_argument("--warp-steps", type=int, help=("warp steps for row_flow_v3"))
    parser.add_argument("--convergence", "-c", type=float, default=0.5,
                        help=("(normalized) distance of convergence plane(screen position). 0-1 is reasonable value"))
    parser.add_argument("--update", action="store_true",
                        help="force update midas models from torch hub")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="process all subdirectories")
    parser.add_argument("--resume", action="store_true",
                        help="skip processing when the output file already exists")
    parser.add_argument("--batch-size", type=int, default=2, choices=[Range(1, 64)],
                        help="batch size. ignored when --low-vram")
    parser.add_argument("--max-fps", type=float, default=30,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--profile-level", type=str, help="h264 profile level")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value for video. smaller value is higher quality")
    parser.add_argument("--video-bitrate", type=str, default="8M",
                        help="bitrate option for libopenh264")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency"],
                        help="encoder tunings option for video")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="overwrite output files")
    parser.add_argument("--pad", type=float, help="pad_size = round(width * pad) // 2")
    parser.add_argument("--pad-mode", type=str, default="tblr", choices=["tblr", "tb", "lr", "16:9"], help="padding mode")
    parser.add_argument("--depth-model", type=str, default="ZoeD_Any_N",
                        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK",
                                 "Any_S", "Any_B", "Any_L",
                                 "ZoeD_Any_N", "ZoeD_Any_K",
                                 "Any_V2_S", "Any_V2_B", "Any_V2_L",
                                 "Any_V2_N", "Any_V2_K",
                                 "Any_V2_N_S", "Any_V2_N_B", "Any_V2_N_L",
                                 "Any_V2_K_S", "Any_V2_K_B", "Any_V2_K_L",
                                 "Distill_Any_S", "Distill_Any_B", "Distill_Any_L",
                                 "DepthPro", "DepthPro_S",
                                 "VDA_S", "VDA_L", "VDA_Metric",
                                 "VDA_Stream_S", "VDA_Stream_L",
                                 "NULL",
                                 ],
                        help="depth model name")
    parser.add_argument("--remove-bg", action="store_true",
                        help="remove background depth, not recommended for video (DELETED)")
    parser.add_argument("--bg-model", type=str, default="u2net_human_seg",
                        help="rembg model type")
    parser.add_argument("--rotate-left", action="store_true",
                        help="Rotate 90 degrees to the left(counterclockwise)")
    parser.add_argument("--disable-exif-transpose", action="store_true",
                        help="Disable EXIF orientation transpose")
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
    parser.add_argument("--export-depth-only", action="store_true",
                        help=("output only depth image and omits rgb image"))
    parser.add_argument("--export-depth-fit", action="store_true",
                        help=("fit depth image size to rgb image"))
    parser.add_argument("--mapper", type=str,
                        choices=MAPPER_ALL,
                        help=("(re-)mapper function for depth. "
                              "if auto, div_6 for ZoeDepth model, none for DepthAnything/DepthPro model. "
                              "directly using this option is not recommended. "
                              "use --foreground-scale instead."))
    parser.add_argument("--foreground-scale", type=float, choices=[Range(-3.0, 3.0)], default=0,
                        help="foreground scaling level. 0 is disabled")
    parser.add_argument("--mapper-type", type=str, choices=["div", "mul", "shift"], default=None,
                        help="mapper type for foreground scaling level")
    parser.add_argument("--vr180", action="store_true",
                        help="output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true",
                        help="output in Half SBS")
    parser.add_argument("--tb", action="store_true", help="output in Full TopBottom")
    parser.add_argument("--half-tb", action="store_true", help="output in Half TopBottom")

    parser.add_argument("--anaglyph", type=str, nargs="?", default=None, const="dubois",
                        choices=["color", "gray", "half-color", "wimmer", "wimmer2", "dubois", "dubois2"],
                        help="output in anaglyph 3d")
    parser.add_argument("--cross-eyed", action="store_true", help="output for cross-eyed viewing")
    parser.add_argument("--rgbd", action="store_true", help="output in RGBD")
    parser.add_argument("--half-rgbd", action="store_true", help="output in Half RGBD")

    parser.add_argument("--pix-fmt", type=str, default="yuv420p", choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp", "gbrp10le", "gbrp16le"],
                        help="pixel format (video only)")
    parser.add_argument("--tta", action="store_true",
                        help="Use flip augmentation on depth model")
    parser.add_argument("--disable-amp", action="store_true",
                        help="disable AMP for some special reason")
    parser.add_argument("--cuda-stream", action="store_true",
                        help="use multi cuda stream for each thread/device")
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
    parser.add_argument("--resolution", type=int,
                        help="input resolution(small side) for depth model")
    parser.add_argument("--stereo-width", type=int,
                        help="input width for row_flow_v3/row_flow_v2 model")
    parser.add_argument("--ipd-offset", type=float, default=0,
                        help="IPD Offset (width scale %%). 0-10 is reasonable value for Full SBS")
    parser.add_argument("--ema-normalize", action="store_true",
                        help="use min/max moving average to normalize video depth")
    parser.add_argument("--ema-decay", type=float, default=0.75,
                        help="parameter for ema-normalize (0-1). large value makes it smoother")
    parser.add_argument("--ema-buffer", type=int, default=30, help="TODO")
    parser.add_argument("--scene-detect", action="store_true",
                        help=("splitting a scene using shot boundary detection. "
                              "ema and other states will be reset at the boundary of the scene."))
    parser.add_argument("--edge-dilation", type=int, nargs="?", default=None, const=2,
                        help="loop count of edge dilation.")
    parser.add_argument("--depth-aa", action="store_true",
                        help="apply depth antialiasing. ignored for unsupported models")
    parser.add_argument("--max-workers", type=int, default=0, choices=[0, 1, 2, 3, 4, 8, 16],
                        help="max inference worker threads for video processing. 0 is disabled")
    parser.add_argument("--video-format", "-vf", type=str, default="mp4", choices=["mp4", "mkv", "avi"],
                        help="video container format")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "webp", "jpeg"],
                        help="output image format")
    parser.add_argument("--video-codec", "-vc", type=str, default=None, help="video codec")

    parser.add_argument("--metadata", type=str, nargs="?", default=None, const="filename", choices=["filename"],
                        help="Add metadata")
    parser.add_argument("--find-param", type=str, nargs="+",
                        choices=["divergence", "convergence", "foreground-scale", "ipd-offset"],
                        help="outputs results for various parameter combinations")

    # TODO: Change the default value from "unspecified" to "auto"
    parser.add_argument("--colorspace", type=str, default="unspecified",
                        choices=["unspecified", "auto",
                                 "bt709", "bt709-pc", "bt709-tv",
                                 "bt601", "bt601-pc", "bt601-tv",
                                 "bt2020-tv", "bt2020-pq-tv"],
                        help="video colorspace")
    # Deprecated
    parser.add_argument("--zoed-batch-size", type=int,
                        help="Deprecated. Use --batch-size instead")
    parser.add_argument("--zoed-height", type=int,
                        help="Deprecated. Use --resolution instead")

    return parser


def calc_auto_warp_steps(method, divergence, synthetic_view):
    divergence = divergence if synthetic_view == "both" else divergence * 2
    if method == "row_flow_v2" and divergence > ROW_FLOW_V2_MAX_DIVERGENCE:
        return math.ceil(divergence / ROW_FLOW_V2_AUTO_STEP_DIVERGENCE)
    if method in {"row_flow", "row_flow_v3"} and divergence > ROW_FLOW_V3_MAX_DIVERGENCE:
        return math.ceil(divergence / ROW_FLOW_V3_AUTO_STEP_DIVERGENCE)

    return None


def set_state_args(args, stop_event=None, tqdm_fn=None, depth_model=None, suspend_event=None):
    if depth_model is None:
        depth_model = create_depth_model(args.depth_model)

    if args.inpaint_model is not None:
        inpaint_model = load_model(args.inpaint_model, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
    else:
        inpaint_model = None

    if args.export_disparity:
        args.export = True
    if args.export_depth_only and not args.export:
        raise ValueError("--export-depth-only must be specified together with --export or --export-disparity")
    if args.export_depth_fit and not args.export:
        raise ValueError("--export-depth-fit must be specified together with --export or --export-disparity")

    if depth_model.get_name() == "VideoDepthAnything":
        if not args.ema_normalize:
            warnings.warn("--ema-normalize is highly recommended for VideoDepthAnything")
        if not args.scene_detect:
            warnings.warn("--scene-detect is highly recommended for VideoDepthAnything")

    if is_video(args.output):
        # replace --video-format when filename is specified
        ext = path.splitext(args.output)[-1]
        if ext == ".mp4":
            args.video_format = "mp4"
        elif ext == ".mkv":
            args.video_format = "mkv"
        elif ext == ".avi":
            args.video_format = "avi"

    args.video_extension = "." + args.video_format
    if args.video_codec is None:
        args.video_codec = VU.get_default_video_codec(args.video_format)

    if not args.profile_level or args.profile_level == "auto":
        args.profile_level = None

    # deprecated options
    if args.zoed_batch_size is not None:
        args.batch_size = args.zoed_batch_size
        warnings.warn("--zoed-batch-size is deprecated. Use --batch-size instead")
    if args.zoed_height is not None:
        args.resolution = args.zoed_height
        warnings.warn("--zoed-height is deprecated. Use --resolution instead")
    if args.remove_bg:
        warnings.warn("--remove-bg is deleted")

    args.state = {
        "stop_event": stop_event,
        "suspend_event": suspend_event,
        "tqdm_fn": tqdm_fn,
        "depth_model": depth_model,
        "inpaint_model": inpaint_model,
        "device": create_device(args.gpu),
        "devices": [create_device(gpu_id) for gpu_id in args.gpu],
    }

    gc_collect()

    return args


def export_main(args):
    if is_text(args.input):
        raise NotImplementedError("--export with text format input is not supported")

    depth_model = args.state["depth_model"]

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            if depth_model.is_image_supported():
                export_images(args.input, args.output, args)
                gc_collect()
            if depth_model.is_video_supported():
                for video_file in VU.list_videos(args.input):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    export_video(video_file, args.output, args)
                    gc_collect()
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                if depth_model.is_image_supported():
                    export_images(input_dir, output_dir, args, title=path.relpath(input_dir, args.input))
                    gc_collect()
                if depth_model.is_video_supported():
                    for video_file in VU.list_videos(input_dir):
                        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                            return args
                        export_video(video_file, output_dir, args)
                        gc_collect()

    elif is_image(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not depth_model.is_image_supported():
            raise ValueError(f"{args.depth_model} does not support image input")
        export_images(args.input, args.output, args)
    elif is_video(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not depth_model.is_video_supported():
            raise ValueError(f"{args.depth_model} not support video input")
        export_video(args.input, args.output, args)
    else:
        raise ValueError("Unrecognized file type")


def is_yaml(filename):
    return path.splitext(filename)[-1].lower() in {".yaml", ".yml"}


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)
    assert sum([1 for flag in (args.half_sbs, args.vr180, args.anaglyph, args.tb, args.half_tb, args.cross_eyed, args.half_rgbd, args.rgbd) if flag]) < 2

    if len(args.gpu) > 1 and len(args.gpu) > args.max_workers:
        # For GPU round-robin on thread pool
        args.max_workers = len(args.gpu)

    if args.warp_steps is None:
        args.warp_steps = calc_auto_warp_steps(method=args.method, divergence=args.divergence,
                                               synthetic_view=args.synthetic_view)

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

    if args.export and is_yaml(args.input):
        raise ValueError("YAML file input does not support --export")

    if args.tune and args.video_codec == "libx265":
        if len(args.tune) != 1:
            raise ValueError("libx265 does not support multiple --tune options.\n"
                             f"tune={','.join(args.tune)}")
        if args.tune[0] in {"film", "stillimage"}:
            raise ValueError(f"libx265 does not support --tune {args.tune[0]}\n"
                             "available options: grain,animation,psnr,zerolatency,fastdecode")

    assert args.state["depth_model"] is not None
    depth_model = args.state["depth_model"]
    if args.update:
        depth_model.force_update()

    if args.edge_dilation is None:
        if depth_model.get_name() in {"DepthAnything", "DepthPro", "VideoDepthAnything"} and not (args.rgbd or args.half_rgbd):
            # TODO: This may not be a sensible choice
            args.edge_dilation = 2
        else:
            args.edge_dilation = 0

    if not is_yaml(args.input):
        if not depth_model.loaded():
            depth_model.load(gpu=args.gpu, resolution=args.resolution)

        is_metric = depth_model.is_metric()
        args.mapper = resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                          metric_depth=is_metric,
                                          mapper_type=args.mapper_type)
    else:
        depth_model = None
        # specified args.mapper never used in process_config_*
        args.mapper = "none"

    if args.export:
        export_main(args)
        return args

    side_model = create_stereo_model(
        args.method,
        divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
        device_id=args.gpu[0]
    )
    if side_model is not None and len(args.gpu) > 1:
        side_model = DeviceSwitchInference(side_model, device_ids=args.gpu)

    if args.find_param:
        assert is_image(args.input) and (path.isdir(args.output) or not path.exists(args.output))
        find_param(args, depth_model, side_model)
        return args

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            if depth_model.is_image_supported():
                image_files = ImageLoader.listdir(args.input)
                process_images(image_files, args.output, args, depth_model, side_model, title="Images")
                gc_collect()
            if depth_model.is_video_supported():
                for video_file in VU.list_videos(args.input):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    process_video(video_file, args.output, args, depth_model, side_model)
                    gc_collect()
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                if depth_model.is_image_supported():
                    image_files = ImageLoader.listdir(input_dir)
                    if image_files:
                        process_images(image_files, output_dir, args, depth_model, side_model,
                                       title=path.relpath(input_dir, args.input))
                        gc_collect()
                if depth_model.is_video_supported():
                    for video_file in VU.list_videos(input_dir):
                        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                            return args
                        process_video(video_file, output_dir, args, depth_model, side_model)
                        gc_collect()

    elif is_yaml(args.input):
        config = export_config.ExportConfig.load(args.input)
        if config.type == export_config.VIDEO_TYPE:
            process_config_video(config, args, side_model)
        if config.type == export_config.IMAGE_TYPE:
            process_config_images(config, args, side_model)
    elif is_text(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        files = []
        with open(args.input, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line.startswith("#"):
                    files.append(line.strip())
        if depth_model.is_image_supported():
            image_files = [f for f in files if is_image(f)]
            process_images(image_files, args.output, args, depth_model, side_model, title="Images")
        if depth_model.is_video_supported():
            video_files = [f for f in files if is_video(f)]
            for video_file in video_files:
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return args
                process_video(video_file, args.output, args, depth_model, side_model)
                gc_collect()
    elif is_video(args.input):
        if not depth_model.is_video_supported():
            raise ValueError(f"{args.depth_model} does not support video input")
        process_video(args.input, args.output, args, depth_model, side_model)
    elif is_image(args.input):
        if not depth_model.is_image_supported():
            raise ValueError(f"{args.depth_model} does not support image input")
        if is_output_dir(args.output):
            os.makedirs(args.output, exist_ok=True)
            output_filename = path.join(
                args.output,
                make_output_filename(args.input, args, video=False))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb", exif_transpose=not args.disable_exif_transpose)
        im = TF.to_tensor(im).to(args.state["device"])
        output = process_image(im, args, depth_model, side_model)
        output = to_pil_image(output)
        make_parent_dir(output_filename)
        output.save(output_filename)
    else:
        raise ValueError("Unrecognized file type")

    return args


def find_param(args, depth_model, side_model):
    im, _ = load_image_simple(args.input, color="rgb")
    if im is None:
        raise RuntimeError(f"{args.input} cannot be loadded")
    im = TF.to_tensor(im).to(args.state["device"])

    args.metadata = "filename"
    os.makedirs(args.output, exist_ok=True)
    if args.method == "forward_fill":
        divergence_cond = range(1, 10 + 1) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(-2, 2, 0.25) if "convergence" in args.find_param else [args.convergence]
    else:
        max_divegence = 10 if args.method.startswith("mlbw_") else 5
        divergence_cond = range(1, max_divegence + 1) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(0, 1, 0.25) if "convergence" in args.find_param else [args.convergence]

    foreground_scale_cond = range(0, 3 + 1) if "foreground-scale" in args.find_param else [args.foreground_scale]
    ipd_offset_cond = range(0, 5 + 1) if "ipd-offset" in args.find_param else [args.ipd_offset]
    is_metric = depth_model.is_metric()

    params = []
    for divergence in divergence_cond:
        for convergence in convergence_cond:
            for foreground_scale in foreground_scale_cond:
                for ipd_offset in ipd_offset_cond:
                    params.append((divergence, convergence, foreground_scale, ipd_offset))

    for divergence, convergence, foreground_scale, ipd_offset in tqdm(params, ncols=80):
        args.divergence = float(divergence)
        args.convergence = float(convergence)
        args.ipd_offset = ipd_offset
        args.foreground_scale = foreground_scale
        args.mapper = resolve_mapper_name(mapper=None, foreground_scale=args.foreground_scale,
                                          metric_depth=is_metric,
                                          mapper_type=args.mapper_type)

        output_filename = path.join(
            args.output,
            make_output_filename("param.png", args, video=False))
        output = process_image(im, args, depth_model, side_model)
        output = to_pil_image(output)
        output.save(output_filename)
