import os
from os import path
import warnings
import torch
from PIL import Image
import argparse
import csv
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from nunif.logger import logger
from nunif.device import create_device, mps_is_available, xpu_is_available
from nunif.utils.image_loader import ImageLoader
from nunif.utils.filename import set_image_ext
from nunif.utils.rgb_noise import rgb_noise_like, apply_rgb_noise
from nunif.utils import video as VU
from nunif.utils.ui import (
    is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir)
from .utils import Waifu2x
from .download_models import main as download_main


MODEL_DIR = path.join(path.dirname(path.abspath(__file__)), "pretrained_models")
DEFAULT_ART_MODEL_DIR = path.join(MODEL_DIR, "swin_unet", "art")
DEFAULT_ART_SCAN_MODEL_DIR = path.join(MODEL_DIR, "swin_unet", "art_scan")
DEFAULT_PHOTO_MODEL_DIR = path.join(MODEL_DIR, "swin_unet", "photo")


SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, args, video=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})

    if video:
        return basename + args.video_extension
    else:
        return set_image_ext(basename, args.format)


@torch.inference_mode()
def process_image(ctx, im, meta, args):
    if args.rotate_left:
        im = im.transpose(Image.Transpose.ROTATE_90)
    elif args.rotate_right:
        im = im.transpose(Image.Transpose.ROTATE_270)

    rgb, alpha = IL.to_tensor(im, return_alpha=True)
    if rgb.shape[0] == 1:
        rgb = rgb.repeat(3, 1, 1)

    rgb, alpha = ctx.convert(
        rgb, alpha, args.method, args.noise_level,
        args.tile_size, args.batch_size,
        args.tta, enable_amp=not args.disable_amp,
        output_device=args.state["device"],
    )
    if args.grain:
        noise = rgb_noise_like(rgb)
        # grain_strength = 1/2 for image
        rgb = apply_rgb_noise(rgb, noise, strength=args.grain_strength * 0.5)

    rgb = rgb.to("cpu")

    if args.depth is not None and args.image_lib == "wand":
        meta["depth"] = args.depth
    depth = meta["depth"] if "depth" in meta and meta["depth"] is not None else 8
    if args.grayscale:
        meta["grayscale"] = True

    return IL.to_image(rgb, alpha, depth=depth)


def process_images(ctx, files, output_dir, args, title=None):
    os.makedirs(output_dir, exist_ok=True)
    loader = ImageLoader(files=files, max_queue_size=128,
                         load_func=IL.load_image,
                         load_func_kwargs={"color": "rgb", "keep_alpha": True,
                                           "exif_transpose": not args.disable_exif_transpose})
    futures = []
    with PoolExecutor(max_workers=cpu_count() // 2 or 1) as pool:
        tqdm_fn = args.state["tqdm_fn"] or tqdm
        pbar = tqdm_fn(ncols=80, total=len(files), desc=title)
        for im, meta in loader:
            output_filename = path.join(
                output_dir,
                make_output_filename(meta["filename"], args, video=False))
            if args.resume and path.exists(output_filename):
                continue
            output = process_image(ctx, im, meta, args)
            futures.append(pool.submit(
                IL.save_image, output,
                filename=output_filename,
                meta=meta, format=args.format))
            pbar.update(1)
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                break
        for f in futures:
            f.result()
        pbar.close()


def process_video(ctx, input_filename, output_path, args):
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)
    if args.compile:
        ctx.compile()

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps

        if args.video_codec in {"libx264", "libx265", "hevc_nvenc", "h264_nvenc"}:
            options = {"preset": args.preset, "crf": str(args.crf)}
            tune = []
            if args.tune:
                tune += args.tune
            tune = set(tune)
            if tune:
                options["tune"] = ",".join(tune)

            if args.profile_level:
                options["level"] = str(int(float(args.profile_level) * 10))

            if args.video_codec == "libx265":
                x265_params = ["log-level=warning", "high-tier=enabled"]
                if args.profile_level:
                    x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")
                options["x265-params"] = ":".join(x265_params)
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

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            options=options,
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        )

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            return None

        rgb = VU.to_tensor(frame, device=args.state["device"])
        if args.rotate_left:
            rgb = torch.rot90(rgb, 1, (-2, -1))
        elif args.rotate_right:
            rgb = torch.rot90(rgb, 3, (-2, -1))

        output, _ = ctx.convert(
            rgb, None, args.method, args.noise_level,
            args.tile_size, args.batch_size,
            args.tta, enable_amp=not args.disable_amp,
            output_device=rgb.device)
        if args.grain:
            noise = rgb_noise_like(output)
            if noise.shape != args.state["noise_buffer"].shape:
                args.state["noise_buffer"].resize_(noise.shape)
                args.state["noise_buffer"].copy_(noise)
            else:
                args.state["noise_buffer"].mul_((1.0 - args.grain_speed))
                args.state["noise_buffer"].add_(noise.mul_(args.grain_speed))
            output = apply_rgb_noise(output, args.state["noise_buffer"], strength=args.grain_strength)

        return VU.to_frame(output, use_16bit=use_16bit)

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(input_filename, args, video=True))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

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


def load_files(txt):
    files = []
    with open(txt, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
    return files


def create_parser(required_true=True):
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=str, help="model dir")
    parser.add_argument("--noise-level", "-n", type=int, default=0, choices=[0, 1, 2, 3], help="noise level")
    parser.add_argument("--method", "-m", type=str,
                        choices=["scale4x", "scale2x",
                                 "noise_scale4x", "noise_scale2x",
                                 "scale", "noise", "noise_scale"],
                        default="noise_scale", help="method")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device ids. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="minibatch_size")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="tile size for tiled render")
    parser.add_argument("--output", "-o", type=str, required=required_true,
                        help="output file or directory")
    parser.add_argument("--input", "-i", type=str, required=required_true,
                        help="input file or directory. (*.txt, *.csv) for image list")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--compile", action="store_true", help="compile model if possible")
    parser.add_argument("--image-lib", type=str, choices=["pil", "wand"], default="pil",
                        help="image library to encode/decode images")
    parser.add_argument("--depth", type=int,
                        help="bit-depth of output image. enabled only with `--image-lib wand`")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "webp", "jpeg"],
                        help="output image format")
    parser.add_argument("--style", type=str, choices=["art", "photo", "scan", "art_scan"],
                        help=("style for default model (art/scan/photo). "
                              "Ignored when --model-dir option is specified."))
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert to grayscale format")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="process all subdirectories")
    parser.add_argument("--resume", action="store_true",
                        help="skip processing when the output file already exists")

    parser.add_argument("--video-format", "-vf", type=str, default="mp4", choices=["mp4", "mkv", "avi"],
                        help="video container format")
    parser.add_argument("--video-codec", "-vc", type=str, default=None, help="video codec")
    parser.add_argument("--max-fps", type=float, default=128,
                        help="max framerate. output fps = min(fps, --max-fps) (video only)")
    parser.add_argument("--profile-level", type=str, help="h264 profile level")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value. smaller value is higher quality (video only)")
    parser.add_argument("--video-bitrate", type=str, default="8M",
                        help="bitrate option for libopenh264")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="encoder preset option (video only)")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency"],
                        help="encoder tunings option (video only)")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p", choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp", "gbrp10le", "gbrp16le"],
                        help=("pixel format (video only)"))
    parser.add_argument("--colorspace", type=str, default="unspecified",
                        choices=["unspecified", "auto",
                                 "bt709", "bt709-pc", "bt709-tv",
                                 "bt601", "bt601-pc", "bt601-tv",
                                 "bt2020-tv", "bt2020-pq-tv"],
                        help="video colorspace")

    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="overwrite output files (video only)")
    parser.add_argument("--rotate-left", action="store_true",
                        help="Rotate 90 degrees to the left(counterclockwise) (video only)")
    parser.add_argument("--rotate-right", action="store_true",
                        help="Rotate 90 degrees to the right(clockwise) (video only)")
    parser.add_argument("--disable-exif-transpose", action="store_true",
                        help="Disable EXIF orientation transpose")
    parser.add_argument("--vf", type=str, default="",
                        help="video filter options for ffmpeg. (video only)")
    parser.add_argument("--grain", action="store_true",
                        help=("add noise after denosing"))
    parser.add_argument("--grain-strength", type=float, default=0.2,
                        help=("noise strength"))
    parser.add_argument("--grain-speed", type=float, default=0.8,
                        help=("noise update speed (video only)"))
    parser.add_argument("--start-time", type=str,
                        help="set the start time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--end-time", type=str,
                        help="set the end time offset for video. hh:mm:ss or mm:ss format")

    # Deprecated
    parser.add_argument("--disable-compile", action="store_true", help="disable torch.compile(). Deprecated")

    return parser


def set_state_args(args, stop_event=None, tqdm_fn=None):
    device = create_device(args.gpu)

    if args.disable_compile:
        warnings.warn("--disable-compile is deprecated. It is disabled by default. Use --compile instead")

    if not args.profile_level or args.profile_level == "auto":
        args.profile_level = None

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

    args.state = {
        "device": device,
        "noise_buffer": torch.zeros((1, 1, 1), dtype=torch.float32, device=device),
        "stop_event": stop_event,
        "tqdm_fn": tqdm_fn,
    }
    return args


def waifu2x_main(args):
    logger.debug(f"waifu2x.cli.main: {str(args)}")
    if args.image_lib == "wand":
        global IL
        from nunif.utils import wand_io as IL
    else:
        global IL
        from nunif.utils import pil_io as IL

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

    # alias for typo
    if args.method == "scale2x":
        args.method = "scale"
    elif args.method == "noise_scale2x":
        args.method = "noise_scale"

    # download models
    pretrained_model_dir = path.join(path.dirname(__file__), "pretrained_models")
    if not path.exists(pretrained_model_dir):
        download_main()

    # main
    if args.model_dir is None:
        if args.style == "photo":
            model_dir = DEFAULT_PHOTO_MODEL_DIR
        elif args.style in {"scan", "art_scan"}:
            model_dir = DEFAULT_ART_SCAN_MODEL_DIR
        else:
            model_dir = DEFAULT_ART_MODEL_DIR
    else:
        model_dir = args.model_dir

    ctx = Waifu2x(model_dir=model_dir, gpus=args.gpu)
    ctx.load_model(args.method, args.noise_level)

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            image_files = ImageLoader.listdir(args.input)
            if image_files:
                process_images(ctx, image_files, args.output, args, title="Images")
            for video_file in VU.list_videos(args.input):
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return
                process_video(ctx, video_file, args.output, args)
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                image_files = ImageLoader.listdir(input_dir)
                if image_files:
                    process_images(ctx, image_files, output_dir, args,
                                   title=path.relpath(input_dir, args.input))
                for video_file in VU.list_videos(input_dir):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return
                    process_video(ctx, video_file, output_dir, args)
    elif is_text(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        files = load_files(args.input)
        image_files = [f for f in files if is_image(f)]
        if image_files:
            process_images(ctx, image_files, args.output, args, title="Images")
        video_files = [f for f in files if is_video(f)]
        for video_file in video_files:
            process_video(ctx, video_file, args.output, args)
    elif is_video(args.input):
        process_video(ctx, args.input, args.output, args)
    elif is_image(args.input):
        if is_output_dir(args.output):
            os.makedirs(args.output, exist_ok=True)
            fmt = args.format
            output_filename = path.join(
                args.output,
                make_output_filename(args.input, args, video=False))
        else:
            _, ext = path.splitext(args.output)
            fmt = ext.lower()[1:]
            if fmt not in {"png", "webp", "jpeg", "jpg"}:
                raise ValueError(f"Unable to recognize image extension: {fmt}")
            output_filename = args.output
        if args.resume and path.exists(output_filename):
            return
        im, meta = IL.load_image(args.input, color="rgb", keep_alpha=True, exif_transpose=not args.disable_exif_transpose)
        output = process_image(ctx, im, meta, args)
        make_parent_dir(output_filename)
        IL.save_image(output, filename=output_filename, meta=meta, format=fmt)
