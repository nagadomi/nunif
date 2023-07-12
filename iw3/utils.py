import os
from os import path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import math
from tqdm import tqdm
import mimetypes
from PIL import Image, ImageDraw
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.utils.seam_blending import SeamBlending
from nunif.models import load_model, get_model_device
from nunif.device import create_device
import nunif.utils.video as VU
from nunif.utils.ui import HiddenPrints


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
    divergence_feat = torch.full_like(depth, divergence_value)
    convergence_feat = torch.full_like(depth, convergence_value)
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    grid = torch.stack((mesh_x, mesh_y), 2)
    grid = grid.permute(2, 0, 1)  # CHW

    return torch.cat([
        c,
        depth.unsqueeze(0),
        divergence_feat.unsqueeze(0),
        convergence_feat.unsqueeze(0),
        grid,
    ], dim=0)


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True):
    x = TF.to_tensor(im).unsqueeze(0).to(model.device)
    if flip_aug:
        x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)

    pad_h = int((x.shape[2] * 0.5) ** 0.5 * 3)
    pad_w = int((x.shape[3] * 0.5) ** 0.5 * 3)
    x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")

    out = model(x)['metric_depth']
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]),
                            mode="bicubic", align_corners=False)
    if pad_h > 0:
        out = out[:, :, pad_h:-pad_h, :]
    if pad_w > 0:
        out = out[:, :, :, pad_w:-pad_w]
    if flip_aug:
        z = (out[0] + torch.flip(out[1], dims=[2])) * 128
    else:
        z = out[0] * 256
    return z.cpu().to(torch.int16)


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
    cos_elevation = torch.cos(elevation)
    x = cos_elevation * torch.sin(azimuth)
    y = torch.sin(elevation)
    z = cos_elevation * torch.cos(azimuth)
    mesh_x = 0.6666 * x / z
    mesh_y = 0.6666 * y / z
    grid = torch.stack((mesh_x, mesh_y), 2)
    z = F.grid_sample(c.unsqueeze(0),
                      grid.unsqueeze(0),
                      mode="bicubic", padding_mode="zeros",
                      align_corners=True).squeeze(0)
    z = torch.clamp(z, 0, 1)

    return z


def is_image(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("image")


def is_video(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("video")


def is_text(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("text")


def is_output_dir(filename):
    return path.isdir(filename) or "." not in path.basename(filename)


def apply_divergence_grid_sample(c, depth, divergence, convergence,
                                 shift, device="cpu"):
    depth = depth.to(device)
    c = c.to(device)
    w, h = c.shape[2], c.shape[1]
    shift_size = (-shift * divergence * 0.01)
    index_shift = depth * shift_size - (shift_size * convergence)
    mesh_y, mesh_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij")
    mesh_x = mesh_x - index_shift
    grid = torch.stack((mesh_x, mesh_y), 2)
    z = F.grid_sample(c.unsqueeze(0), grid.unsqueeze(0),
                      mode="bicubic", padding_mode="border", align_corners=True)
    z = z.squeeze(0)
    z = torch.clamp(z, 0., 1.)
    return z.cpu()


def apply_divergence_nn(model, c, depth, divergence, convergence,
                        mapper, shift, batch_size=64):
    device = get_model_device(model)
    enable_amp = "cuda" in str(device)
    image_width = c.shape[2]
    depth_min, depth_max = depth.min(), depth.max()
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


def load_depth_model(model_type="ZoeD_N", gpu=0):
    with HiddenPrints():
        model = torch.hub.load("isl-org/ZoeDepth:main", model_type, config_mode="infer",
                               pretrained=True, verbose=False)
    device = create_device(gpu)
    model = model.to(device).eval()
    return model


def force_update_midas_model():
    # See https://github.com/isl-org/ZoeDepth/blob/main/hubconf.py
    # Triggers fresh download of MiDaS repo
    torch.hub.help("isl-org/MiDaS", "DPT_BEiT_L_384", force_reload=True)


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
SBS_SUFFIX = "_LRF"
VR180_SUFFIX = "_180x180_LR"


# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, video=False, vr180=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    auto_detect_suffix = VR180_SUFFIX if vr180 else SBS_SUFFIX
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


def process_image_impl(im, args, depth_model, side_model):
    with torch.inference_mode():
        if args.rotate_left:
            im = im.transpose(Image.Transpose.ROTATE_90)
        elif args.rotate_right:
            im = im.transpose(Image.Transpose.ROTATE_270)
        im_org = TF.to_tensor(im)
        if args.bg_session is not None:
            im = remove_bg_from_image(im, args.bg_session)
        if args.disable_zoedepth_batch and args.tta:
            depth = TF.to_tensor(depth_model.infer_pil(im, output_type="pil"))
        else:
            depth = batch_infer(depth_model, im, flip_aug=args.tta)
        if args.method == "grid_sample":
            depth = normalize_depth(depth.squeeze(0))
            depth = get_mapper(args.mapper)(depth)
            left_eye = apply_divergence_grid_sample(
                im_org, depth,
                args.divergence, convergence=args.convergence, shift=-1,
                device=depth_model.device)
            right_eye = apply_divergence_grid_sample(
                im_org, depth,
                args.divergence, convergence=args.convergence, shift=1,
                device=depth_model.device)
        else:
            left_eye = apply_divergence_nn(side_model, im_org, depth,
                                           args.divergence, args.convergence,
                                           args.mapper, shift=-1, batch_size=args.batch_size)
            right_eye = apply_divergence_nn(side_model, im_org, depth,
                                            args.divergence, args.convergence,
                                            args.mapper, shift=1, batch_size=args.batch_size)
        if args.pad is not None:
            pad_h = int(left_eye.shape[1] * args.pad) // 2
            pad_w = int(left_eye.shape[2] * args.pad) // 2
            left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
            right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
        if args.vr180:
            left_eye = equirectangular_projection(left_eye, device=depth_model.device)
            right_eye = equirectangular_projection(right_eye, device=depth_model.device)
        sbs = torch.cat([left_eye, right_eye], dim=2)
        sbs = TF.to_pil_image(sbs)
        return sbs


def generate_depth_debug(im, args, depth_model, side_model):
    with torch.inference_mode():
        if args.rotate_left:
            im = im.transpose(Image.Transpose.ROTATE_90)
        elif args.rotate_right:
            im = im.transpose(Image.Transpose.ROTATE_270)
        if args.bg_session is not None:
            im = remove_bg_from_image(im, args.bg_session)
        if args.disable_zoedepth_batch:
            depth = TF.to_tensor(depth_model.infer_pil(im, output_type="pil"))
        else:
            depth = batch_infer(depth_model, im, flip_aug=args.tta)
        depth = depth.float()
        min_depth, max_depth = depth.min(), depth.max()
        mean_depth, std_depth = round(depth.mean().item(), 4), round(depth.std().item(), 4)
        depth = normalize_depth(depth)
        depth2 = depth ** 2
        out = torch.cat([depth, depth2], dim=2)
        out = TF.to_pil_image(out)
        gc = ImageDraw.Draw(out)
        gc.text((16, 16), f"min={min_depth}\nmax={max_depth}\nmean={mean_depth}\nstd={std_depth}", "gray")

        return out


def process_image(im, args, depth_model, side_model):
    if args.debug_depth:
        return generate_depth_debug(im, args, depth_model, side_model)
    else:
        return process_image_impl(im, args, depth_model, side_model)


def process_images(args, depth_model, side_model):
    os.makedirs(args.output, exist_ok=True)
    files = ImageLoader.listdir(args.input)
    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files))
    with PoolExecutor(max_workers=4) as pool:
        for im, meta in loader:
            filename = meta["filename"]
            output_filename = path.join(args.output, make_output_filename(filename))
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


def process_video_full(args, depth_model, side_model):
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
            options=options
        )

    def frame_callback(frame):
        return frame.from_image(process_image(frame.to_image(), args, depth_model, side_model))

    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_filename = path.join(
            args.output,
            make_output_filename(path.basename(args.input), video=True, vr180=args.vr180))
    else:
        output_filename = args.output

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    VU.process_video(args.input, output_filename,
                     config_callback=config_callback,
                     frame_callback=frame_callback,
                     vf=args.vf,
                     stop_event=args.state["stop_event"],
                     tqdm_fn=args.state["tqdm_fn"])


def process_video_keyframes(args, depth_model, side_model):
    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_dir = path.join(args.output, make_output_filename(path.basename(args.input), video=True))
    else:
        output_dir = args.output
    output_dir = path.join(path.dirname(output_dir), path.splitext(path.basename(output_dir))[0])
    if output_dir.endswith("_LRF"):
        output_dir = output_dir[:-4]
    os.makedirs(output_dir, exist_ok=True)
    with PoolExecutor(max_workers=4) as pool:
        futures = []

        def frame_callback(frame):
            output = process_image(frame.to_image(), args, depth_model, side_model)
            output_filename = path.join(
                output_dir,
                path.basename(output_dir) + "_" + str(frame.index).zfill(8) + SBS_SUFFIX + ".png")
            f = pool.submit(save_image, output, output_filename)
            futures.append(f)
        VU.process_video_keyframes(args.input, frame_callback=frame_callback,
                                   min_interval_sec=args.keyframe_interval,
                                   stop_event=args.state["stop_event"])
        for f in futures:
            f.result()


def process_video(args, depth_model, side_model):
    if args.keyframe:
        process_video_keyframes(args, depth_model, side_model)
    else:
        process_video_full(args, depth_model, side_model)


FLOW_MODEL_PATH = path.join(path.dirname(__file__), "pretrained_models", "row_flow.pth")


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
    parser.add_argument("--gpu", "-g", type=int, default=default_gpu,
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--method", type=str, default="row_flow",
                        choices=["grid_sample", "row_flow"],
                        help="left-right divergence method")
    parser.add_argument("--divergence", "-d", type=float, default=2.0, choices=[Range(0.0, 2.5)],
                        help=("strength of 3D effect"))
    parser.add_argument("--convergence", "-c", type=float, default=0.5, choices=[Range(0.0, 1.0)],
                        help=("(normalized) distance of convergence plane(screen position)"))
    parser.add_argument("--update", action="store_true",
                        help="force update midas models from torch hub")
    parser.add_argument("--resume", action="store_true",
                        help="skip processing when output file is already exist")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for 256x256 tiled input")
    parser.add_argument("--max-fps", type=float, default=30,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value for video. smaller value is higher quality")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo"],
                        help="encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=["zerolatency"],
                        choices=["film", "animation", "grain", "stillimage",
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
    parser.add_argument("--disable-zoedepth-batch", action="store_true",
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
    parser.add_argument("--tta", action="store_true",
                        help="Use flip augmentation on depth model")
    return parser


def set_state_args(args, stop_event=None, tqdm_fn=None):
    args.state = {"stop_event": stop_event, "tqdm_fn": tqdm_fn}
    return args


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)

    if args.update:
        force_update_midas_model()
    if args.remove_bg:
        global rembg
        import rembg
        args.bg_session = rembg.new_session(model_name=args.bg_model)
    else:
        args.bg_session = None

    depth_model = load_depth_model(model_type=args.depth_model, gpu=args.gpu)
    if args.method == "row_flow":
        side_model = load_model(FLOW_MODEL_PATH, device_ids=[args.gpu])[0].eval()
    else:
        side_model = None

    if path.isdir(args.input):
        process_images(args, depth_model, side_model)
    else:
        if is_video(args.input):
            process_video(args, depth_model, side_model)
        else:
            if is_text(args.input):
                files = []
                with open(args.input, mode="r", encoding="utf-8") as f:
                    for line in f.readlines():
                        files.append(line.strip())
            else:
                files = [args.input]
            for input_file in files:
                if is_video(input_file):
                    args.input = input_file
                    process_video(args, depth_model, side_model)
                elif is_image(input_file):
                    if is_output_dir(args.output):
                        os.makedirs(args.output, exist_ok=True)
                        output_filename = path.join(args.output, make_output_filename(input_file))
                    else:
                        output_filename = args.output
                    im, _ = load_image_simple(input_file, color="rgb")
                    output = process_image(im, args, depth_model, side_model)
                    output.save(output_filename)
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    break
    return True
