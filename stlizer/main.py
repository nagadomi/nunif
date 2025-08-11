import torch
import os
from os import path
import argparse
from nunif.device import mps_is_available, xpu_is_available, create_device
from nunif.utils.ui import is_video
import nunif.utils.video as VU
from .multipass_pipeline import (
    calc_scene_weight,
    pass1, pass2, pass3, pass4,
    DEFAULT_RESOLUTION,
)
from .cache import try_load_cache, save_cache


def create_parser(required_true=True):
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--gpu", "-g", type=int, default=default_gpu,
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="base batch size")
    parser.add_argument("--smoothing", type=float, default=2.0, help="seconds to smoothing")
    parser.add_argument("--filter", type=str, default="grad_opt",
                        choices=["gaussian", "savgol", "grad_opt"], help="smoothing filter")

    parser.add_argument("--border", type=str, choices=["black", "outpaint", "crop", "expand", "expand_outpaint"],
                        default="black", help="border padding mode")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="pre-padding ratio for --border=expand|expand_outpaint|crop")
    parser.add_argument("--buffer-decay", type=float, default=0.75,
                        help="buffer decay factor for outpaint|expand_outpaint")

    parser.add_argument("--debug", action="store_true", help="debug output original+stabilized")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="resolution to perform processing")
    parser.add_argument("--iteration", type=int, default=50, help="iteration count of frame transform optimization")
    parser.add_argument("--disable-cache", action="store_true",
                        help="disable pass1-2 cache")

    # video encoding

    parser.add_argument("--max-fps", type=float, default=60.0,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p",
                        choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp", "gbrp10le", "gbrp16le"],
                        help="pixel format")
    parser.add_argument("--profile-level", type=str, help="h264 profile level")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value for video. smaller value is higher quality")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency"],
                        help="encoder tunings option for video")
    parser.add_argument("--vf", type=str, default="",
                        help="video filter options for ffmpeg.")
    parser.add_argument("--video-format", "-vf", type=str, default="mp4", choices=["mp4", "mkv", "avi"],
                        help="video container format")
    parser.add_argument("--video-codec", "-vc", type=str, default=None, help="video codec")

    # TODO: Change the default value from "unspecified" to "auto"
    parser.add_argument("--colorspace", type=str, default="unspecified",
                        choices=["unspecified", "auto",
                                 "bt709", "bt709-pc", "bt709-tv",
                                 "bt601", "bt601-pc", "bt601-tv",
                                 "bt2020-tv", "bt2020-pq-tv"],
                        help="video colorspace")

    return parser


def set_state_args(args, stop_event=None, tqdm_fn=None, suspend_event=None):
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

    device = create_device(args.gpu)
    args.state = {
        "stop_event": stop_event,
        "suspend_event": suspend_event,
        "tqdm_fn": tqdm_fn,
        "device": device,
        "devices": [device],
    }
    return args


def make_output_path(args):
    if path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_dir = args.output
        output_path = path.join(
            output_dir,
            path.splitext(path.basename(args.input))[0] + "_stlizer" + args.video_extension
        )
    else:
        output_dir = path.dirname(args.output)
        output_path = args.output

    return output_path


def stlizer_main(args):
    output_path = make_output_path(args)
    device = args.state["device"]

    cache_data = try_load_cache(args.input, args) if not args.disable_cache else None
    if cache_data is None:
        # detect keypoints and matching
        points1, points2, mean_match_scores, center, resize_scale, fps = pass1(args=args)

        # calculate optical flow (transform)
        transforms = pass2(points1, points2, center, resize_scale, args=args)

        assert len(transforms) == len(mean_match_scores)

        if not args.disable_cache:
            save_cache(args.input, transforms, mean_match_scores, fps, args)
    else:
        transforms = cache_data["transforms"]
        mean_match_scores = cache_data["mean_match_scores"]
        fps = cache_data["fps"]

    # stabilize
    scene_weight = calc_scene_weight(mean_match_scores, device=device)
    shift_x_fix, shift_y_fix, angle_fix = pass3(transforms, scene_weight, fps=fps, args=args)

    # warp encode
    pass4(output_path, shift_x_fix, shift_y_fix, angle_fix, transforms, scene_weight, fps=fps, args=args)
