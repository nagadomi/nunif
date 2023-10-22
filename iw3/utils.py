import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import math
from tqdm import tqdm
from PIL import Image, ImageDraw
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.utils.seam_blending import SeamBlending
from nunif.models import load_model, compile_model
import nunif.utils.video as VU
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir
from nunif.device import create_device
from . import zoedepth_model as ZU


FLOW_MODEL_PATH = path.join(path.dirname(__file__), "pretrained_models", "row_flow_fp32.pth")
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


def make_input_tensor(c, depth16, divergence, convergence,
                      image_width, depth_min=None, depth_max=None,
                      mapper="pow2"):
    w, h = c.shape[2], c.shape[1]
    depth = normalize_depth(depth16.squeeze(0), depth_min, depth_max)
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = torch.full_like(depth, divergence_value, device=c.device)
    convergence_feat = torch.full_like(depth, convergence_value, device=c.device)
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


def softplus01(depth):
    # smooth function of `(depth - 0.5) * 2 if depth > 0.5 else 0`
    return torch.log(1. + torch.exp(depth * 12.0 - 6.)) / 6.0


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
    w, h = c.shape[2], c.shape[1]
    shift_size = (-shift * divergence * 0.01)
    index_shift = depth * shift_size - (shift_size * convergence)
    mesh_y, mesh_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=c.device),
        torch.linspace(-1, 1, w, device=c.device),
        indexing="ij")
    mesh_x = mesh_x - index_shift
    grid = torch.stack((mesh_x, mesh_y), 2)
    z = F.grid_sample(c.unsqueeze(0), grid.unsqueeze(0),
                      mode="bicubic", padding_mode="border", align_corners=True)
    z = z.squeeze(0)
    z = torch.clamp(z, 0., 1.)
    return z.cpu()


def apply_divergence_nn(model, c, depth, divergence, convergence,
                        depth_min, depth_max,
                        mapper, shift, batch_size, enable_amp):
    image_width = c.shape[2]
    if shift > 0:
        c = torch.flip(c, (2,))
        depth = torch.flip(depth, (2,))

    def config_callback(x):
        return 8, x.shape[1], x.shape[2]

    def preprocess_callback(_, pad):
        xx = F.pad(c.unsqueeze(0), pad, mode="replicate").squeeze(0)
        dd = F.pad(depth.float().unsqueeze(0), pad, mode="replicate").squeeze(0)
        return (xx, dd)

    def input_callback(p, i1, i2, j1, j2):
        xx = p[0][:, i1:i2, j1:j2]
        dd = p[1][:, i1:i2, j1:j2]
        return make_input_tensor(
            xx, dd,
            divergence=divergence, convergence=convergence,
            image_width=image_width,
            depth_min=depth_min, depth_max=depth_max,
            mapper=mapper)

    z = SeamBlending.tiled_render(
        c, model, tile_size=256, batch_size=batch_size, enable_amp=enable_amp,
        config_callback=config_callback,
        preprocess_callback=preprocess_callback,
        input_callback=input_callback)
    if shift > 0:
        z = torch.flip(z, (2,))
    return z


def has_rembg_model(model_type):
    return path.exists(path.join(REMBG_MODEL_DIR, f"{model_type}.onnx"))


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF"
HALF_SBS_SUFFIX = "_LR"
VR180_SUFFIX = "_180x180_LR"


# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, video=False, vr180=False, half_sbs=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
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
    im = TF.to_pil_image(im)

    return im


def preprocess_image(im, args):
    if args.rotate_left:
        im = im.transpose(Image.Transpose.ROTATE_90)
    elif args.rotate_right:
        im = im.transpose(Image.Transpose.ROTATE_270)

    w, h = im.size
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

    im_org = TF.to_tensor(im)
    if args.bg_session is not None:
        im = remove_bg_from_image(im, args.bg_session)
        im = TF.to_tensor(im)
    else:
        im = im_org
    return im_org, im


def postprocess_image(depth, im_org, args, side_model, ema=False):
    depth_min, depth_max = depth.min(), depth.max()
    if ema:
        depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)

    if args.method == "grid_sample":
        depth = normalize_depth(depth.squeeze(0), depth_min=depth_min, depth_max=depth_max)
        depth = get_mapper(args.mapper)(depth)
        left_eye = apply_divergence_grid_sample(
            im_org, depth,
            args.divergence, convergence=args.convergence, shift=-1)
        right_eye = apply_divergence_grid_sample(
            im_org, depth,
            args.divergence, convergence=args.convergence, shift=1)
    else:
        left_eye = apply_divergence_nn(side_model, im_org, depth,
                                       args.divergence, args.convergence,
                                       depth_min=depth_min, depth_max=depth_max,
                                       mapper=args.mapper, shift=-1,
                                       batch_size=args.batch_size, enable_amp=not args.disable_amp)
        right_eye = apply_divergence_nn(side_model, im_org, depth,
                                        args.divergence, args.convergence,
                                        depth_min=depth_min, depth_max=depth_max,
                                        mapper=args.mapper, shift=1,
                                        batch_size=args.batch_size, enable_amp=not args.disable_amp)

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

    sbs = torch.cat([left_eye, right_eye], dim=2)
    sbs = torch.clamp(sbs, 0., 1.)
    sbs = TF.to_pil_image(sbs)

    w, h = sbs.size
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
    gc.text((16, 16), f"min={depth_min}\nmax={depth_min}\nmean={mean_depth}\nstd={std_depth}", "gray")

    return out


def process_image(im, args, depth_model, side_model):
    with torch.inference_mode():
        im_org, im = preprocess_image(im, args)
        depth = ZU.batch_infer(depth_model, im, flip_aug=args.tta, low_vram=args.low_vram,
                               int16=False, enable_amp=not args.disable_amp,
                               output_device=args.state["device"],
                               device=args.state["device"])
        if not args.debug_depth:
            return postprocess_image(depth, im_org.to(args.state["device"]),
                                     args, side_model)
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
                make_output_filename(filename, video=False, vr180=args.vr180, half_sbs=args.half_sbs))
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
        side_model = compile_model(side_model)

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
            options=options,
            container_options={"movflags": "+faststart"}
        )

    minibatch_queue = []
    minibatch_size = args.zoed_batch_size // 2 or 1 if args.tta else args.zoed_batch_size
    if args.debug_depth:
        postprocess = lambda depth, x_org, args, side_model, ema: debug_depth_image(depth, args, ema)
    else:
        postprocess = postprocess_image

    @torch.inference_mode()
    def run_minibatch():
        if not minibatch_queue:
            return []
        x_orgs = []
        xs = []
        for im in minibatch_queue:
            x_org, x = preprocess_image(im, args)
            x_orgs.append(x_org)
            xs.append(x.unsqueeze(0))
        minibatch_queue.clear()
        x = torch.cat(xs, dim=0).to(args.state["device"])
        if args.bg_session is None:
            x_orgs = x
        else:
            x_orgs = torch.cat(x_orgs, dim=0).to(args.state["device"])
        depths = ZU.batch_infer(depth_model, x, flip_aug=args.tta, low_vram=args.low_vram,
                                int16=False, enable_amp=not args.disable_amp,
                                output_device=args.state["device"],
                                device=args.state["device"])
        return [VU.from_image(postprocess(depth, x_org,
                                          args, side_model, ema_normalize))
                for depth, x_org in zip(depths, x_orgs)]

    def frame_callback(frame):
        if not args.low_vram:
            if frame is None:
                return run_minibatch()
            minibatch_queue.append(frame.to_image())
            if len(minibatch_queue) >= minibatch_size:
                return run_minibatch()
            else:
                return None
        else:
            if frame is None:
                return None
            return VU.from_image(process_image(frame.to_image(), args, depth_model, side_model))

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), video=True,
                                 vr180=args.vr180, half_sbs=args.half_sbs))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    if ema_normalize:
        args.state["ema"].clear()
    make_parent_dir(output_filename)
    VU.process_video(input_filename, output_filename,
                     config_callback=config_callback,
                     frame_callback=frame_callback,
                     vf=args.vf,
                     stop_event=args.state["stop_event"],
                     tqdm_fn=args.state["tqdm_fn"],
                     title=path.basename(input_filename),
                     start_time=args.start_time,
                     end_time=args.end_time)


def process_video_keyframes(input_filename, output_path, args, depth_model, side_model):
    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(output_path,
                                    make_output_filename(path.basename(input_filename),
                                                         video=True, vr180=args.vr180, half_sbs=args.half_sbs))
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
                        help="batch size for RowFlow model, 256x256 tiled input")
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
                        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
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
    parser.add_argument("--mapper", type=str, default="pow2",
                        choices=["pow2", "softplus", "softplus2", "none"],
                        help="(re-)mapper function for depth")
    parser.add_argument("--vr180", action="store_true",
                        help="output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true",
                        help="output in Half SBS")
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
    args.state = {
        "stop_event": stop_event,
        "tqdm_fn": tqdm_fn,
        "depth_model": depth_model,
        "ema": EMAMinMax(),
        "device": create_device(args.gpu),
    }
    return args


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)
    assert not (args.half_sbs and args.vr180)

    if args.update:
        ZU.force_update_midas()
        ZU.force_update_zoedepth()

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

    if args.remove_bg:
        global rembg
        import rembg
        args.bg_session = rembg.new_session(model_name=args.bg_model)
    else:
        args.bg_session = None

    if args.state["depth_model"] is not None:
        depth_model = args.state["depth_model"]
    else:
        depth_model = ZU.load_model(model_type=args.depth_model, gpu=args.gpu, height=args.zoed_height)
        args.state["depth_model"] = depth_model

    if args.method == "row_flow":
        side_model = load_model(FLOW_MODEL_PATH, device_ids=[args.gpu[0]])[0].eval()
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
                make_output_filename(args.input, video=False, vr180=args.vr180, half_sbs=args.half_sbs))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb")
        output = process_image(im, args, depth_model, side_model)
        make_parent_dir(output_filename)
        output.save(output_filename)

    return args
