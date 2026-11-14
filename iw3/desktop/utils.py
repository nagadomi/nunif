import argparse
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from functools import partial
from os import path
from types import SimpleNamespace
from typing import Any

import torch
import wx  # for mouse pointer
from packaging.version import Version
from torchvision.io import encode_jpeg

from nunif.device import create_device
from nunif.initializer import gc_collect
from nunif.models import compile_model
from nunif.models.data_parallel import DeviceSwitchInference

from .. import models  # noqa
from .. import utils as IW3U
from ..player.server import (
    get_local_address,
    validate_bind_address_and_auth,
)
from ..stereo_model_factory import get_mlbw_divergence_level
from .screenshot_process import (  # noqa
    ScreenshotProcess,
    enum_window_names,
    get_monitor_size_list,
    get_window_rect_by_title,
    is_mss_supported,
)
from .screenshot_process_obs import (  # noqa
    ScreenshotProcessOBS,
    is_obs_supported,
    is_wayland,
    obs_unsupported_reason,
)
from .screenshot_thread_cuda import ScreenshotThreadWCCUDA
from .screenshot_thread_kwcapture import (  # noqa
    ScreenshotThreadKWCapture,
    is_kwcapture_supported,
    kwcapture_unsupported_reason,
)
from .screenshot_thread_pil import ScreenshotThreadPIL
from .streaming_server import StreamingServer, WebXRStreamingServer

WAYLAND_CAPTURE_MISSING_MESSAGE = (
    "Wayland capture backend is not available ('pil' uses XWayland, which is usually black). "
    "Install obs-studio/obs-plugins for 'obs', or 'pip install kwcapture' for 'kwcapture' (KDE Plasma)."
)

LocalViewer: object | None
try:
    from .local_viewer import LocalViewer
except ImportError:
    LocalViewer = None


TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)
TORCH_NUM_THREADS = torch.get_num_threads()

# Thread-safe JPEG encoding lock
_jpeg_encode_lock = threading.Lock()


def init_win32():
    if sys.platform == "win32":
        import ctypes

        try:
            # Fix mouse position when Display Scaling is not 100%
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:  # noqa
            pass

        if sys.version_info < (3, 11):
            try:
                # Change timer/sleep precision for 3.10
                # NOTE: python 3.11 or later has high precision sleep.
                ctypes.windll.winmm.timeBeginPeriod(1)
            except:  # noqa
                pass


def init_num_threads(device_id):
    if device_id < 0:
        # cpu
        torch.set_num_threads(TORCH_NUM_THREADS)
    else:
        torch.set_num_threads(1)


def to_uint8(x):
    return x.mul(255).round_().to(torch.uint8)


def fps_sleep(start_time, fps, resolution=2e-4):
    # currently not in used
    end_diff_time = (1 / fps) - resolution
    while time.perf_counter() - start_time < end_diff_time:
        time.sleep(resolution)


def to_jpeg_data(frame, quality, tick, gpu_jpeg=True):
    # Thread-safe JPEG encoding with lock
    with _jpeg_encode_lock:
        if ENABLE_GPU_JPEG and gpu_jpeg and frame.device.type == "cuda":
            jpeg_data = encode_jpeg(to_uint8(frame), quality=quality).cpu()
        else:
            jpeg_data = encode_jpeg(to_uint8(frame).cpu(), quality=quality)

    jpeg_data = jpeg_data.numpy().tobytes()
    # debug_jpeg_data(frame, jpeg_data)
    return (jpeg_data, tick)


def debug_jpeg_data(frame, jpeg_data):
    from torchvision.io import decode_jpeg

    jpeg_data = torch.tensor(list(jpeg_data), dtype=torch.uint8)
    try:
        decodec_frame = decode_jpeg(jpeg_data).cpu() / 255.0
        diff = (frame.cpu() - decodec_frame).abs().mean()
        print(diff)
    except RuntimeError as e:
        print(e)


def add_desktop_stream_args(parser: argparse.ArgumentParser, default_port: int = 1303) -> argparse.ArgumentParser:
    local_address = get_local_address()
    parser.add_argument("--port", type=int, default=default_port, help="HTTP listen port")
    parser.add_argument("--bind-addr", type=str, default=local_address, help="HTTP listen address")
    parser.add_argument("--user", type=str, help="HTTP Basic Authentication username")
    parser.add_argument("--password", type=str, help="HTTP Basic Authentication password")
    parser.add_argument("--stream-fps", type=int, default=30, help="Streaming FPS")
    parser.add_argument("--stream-height", type=int, default=1080, help="Streaming screen resolution")
    parser.add_argument("--stream-quality", type=int, default=90, help="Streaming JPEG quality")
    parser.add_argument("--full-sbs", action="store_true", help="Use Full SBS for Pico4")
    parser.add_argument(
        "--screenshot",
        type=str,
        default="pil",
        choices=["pil", "mss", "wc_mp", "wc_cuda", "kwcapture", "obs"],
        help="Screenshot method",
    )
    parser.add_argument("--gpu-jpeg", action="store_true", help="Use GPU JPEG Encoder")
    parser.add_argument("--monitor-index", type=int, default=0, help="monitor_index for wc_mp. 0 origin. 0 = monitor 1")
    parser.add_argument(
        "--window-name",
        type=str,
        help=("target window name for wc_mp. When this is specified, --monitor-index is ignored"),
    )
    parser.add_argument("--crop-top", type=int, default=0, help="Crop pixels from top when using --window-name")
    parser.add_argument("--crop-left", type=int, default=0, help="Crop pixels from left when using --window-name")
    parser.add_argument("--crop-right", type=int, default=0, help="Crop pixels from right when using --window-name")
    parser.add_argument("--crop-bottom", type=int, default=0, help="Crop pixels from bottom when using --window-name")
    parser.add_argument(
        "--disable-draw-cursor", action="store_true", help="Disables drawing the cursor on the output stream"
    )

    parser.set_defaults(
        input="dummy",
        output="dummy",
        depth_model="Any_V2_S",
        divergence=1.0,
        convergence=1.0,
        ema_normalize=True,
    )
    return parser


def create_parser():
    parser = IW3U.create_parser(required_true=False)
    parser.add_argument("--local-viewer", action="store_true", help="Use Local Viewer instead of Web Streaming")
    parser.add_argument(
        "--legacy-stream",
        action="store_true",
        help="Use legacy HTTP MJPEG streaming server instead of WebXR HTTPS player",
    )
    parser.add_argument(
        "--uncap-fps",
        action="store_true",
        help="Allows LocalViewer to render at frame rates beyond the display's refresh rate",
    )
    parser.add_argument(
        "--fit-window",
        action="store_true",
        help="Stretch image to fit the Local Viewer window ignoring aspect ratio",
    )
    add_desktop_stream_args(parser, default_port=1303)
    return parser


def set_state_args(args, args_lock=None, stop_event=None, fps_event=None, depth_model=None):
    IW3U.set_state_args(args, stop_event=stop_event, depth_model=depth_model)
    args.state["fps_event"] = fps_event
    args.state["args_lock"] = args_lock if args_lock is not None else threading.Lock()
    args.state.setdefault("initial_mapper", getattr(args, "mapper", None))
    args.state.setdefault("mapper_overridden", False)
    if args.edge_dilation is None or (isinstance(args.edge_dilation, list) and len(args.edge_dilation) == 0):
        args.edge_dilation = 2


def test_output_size(size, args, depth_model, side_model):
    frame = torch.zeros((3, *size), dtype=torch.float32).to(args.state["device"])
    sbs = IW3U.process_image(frame, args, depth_model, side_model, autocrop_uncrop=True)
    depth_model.reset()
    return sbs.shape[-2:]


def try_switch_mlbw_model(old_divergence, new_divergence, old_side_model, args):
    assert args.method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}

    old_level = get_mlbw_divergence_level(old_divergence)
    new_level = get_mlbw_divergence_level(new_divergence)

    # print("switch_mlbw_model", old_level, new_level, old_divergence, new_divergence)

    if old_level == new_level:
        return old_side_model

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    new_side_model = IW3U.create_stereo_model(
        args.method,
        divergence=new_divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
        device_id=args.gpu[0],
    )
    if args.compile and not isinstance(new_side_model, DeviceSwitchInference):
        new_side_model = compile_model(new_side_model, args.state["device"])
        gc_collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return new_side_model


def resolve_player_stereo_format(args) -> str:
    if getattr(args, "vr180", False):
        return "SBS_180"
    if getattr(args, "full_sbs", False) or getattr(args, "rgbd", False):
        return "SBS_FULL"
    if getattr(args, "half_sbs", False) or getattr(args, "half_rgbd", False):
        return "SBS_HALF"
    if getattr(args, "cross_eyed", False):
        return "SBS_FULL_CROSS"
    if getattr(args, "tb", False):
        return "TB_FULL"
    if getattr(args, "half_tb", False):
        return "TB_HALF"
    if getattr(args, "anaglyph", None):
        return "FLAT"
    return "SBS_HALF"


def ensure_default_stereo_format(args: Any) -> None:
    if not any(
        [
            args.full_sbs,
            args.tb,
            args.half_tb,
            args.cross_eyed,
            args.rgbd,
            args.half_rgbd,
            args.anaglyph,
            args.vr180,
        ]
    ):
        args.half_sbs = True


def validate_screenshot_args(args: Any) -> Callable[..., Any]:
    if args.screenshot == "pil":
        screenshot_factory: Callable[..., Any] = ScreenshotThreadPIL
    elif args.screenshot == "mss":
        if not is_mss_supported():
            raise ValueError("mss is not supported on Wayland")
        screenshot_factory = partial(ScreenshotProcess, backend="mss")
    elif args.screenshot == "wc_mp":
        screenshot_factory = partial(ScreenshotProcess, backend="windows_capture")
    elif args.screenshot == "wc_cuda":

        def screenshot_factory(*f_args: Any, **f_kwargs: Any) -> Any:
            return ScreenshotThreadWCCUDA(*f_args, **f_kwargs)
    elif args.screenshot == "kwcapture":
        reason = kwcapture_unsupported_reason()
        if reason is not None:
            raise ValueError(f"--screenshot kwcapture: {reason}")
        screenshot_factory = ScreenshotThreadKWCapture
    elif args.screenshot == "obs":
        reason = obs_unsupported_reason()
        if reason is not None:
            raise ValueError(f"--screenshot obs: {reason}")
        screenshot_factory = ScreenshotProcessOBS
    else:
        raise ValueError(f"Unsupported screenshot backend: {args.screenshot}")

    if args.screenshot == "pil" and is_wayland():
        if is_obs_supported() or is_kwcapture_supported():
            print(
                "Warning: 'pil' screenshot on Wayland captures the XWayland screen (usually "
                "black). Use --screenshot obs or --screenshot kwcapture.",
                file=sys.stderr,
            )
        else:
            print(f"Warning: {WAYLAND_CAPTURE_MISSING_MESSAGE}", file=sys.stderr)

    if args.screenshot in {"pil", "obs"} and args.monitor_index != 0:
        raise RuntimeError(f"{args.screenshot} does not support monitor_index={args.monitor_index}")
    if args.screenshot in {"pil", "obs"} and args.window_name:
        raise RuntimeError(f"{args.screenshot} does not support --window-name option")

    return screenshot_factory


def get_runtime_stream_settings(args: Any) -> dict[str, Any]:
    state = getattr(args, "state", None)
    args_lock = state.get("args_lock") if isinstance(state, dict) else None

    def _read() -> dict[str, Any]:
        edge_dilation = (
            int(args.edge_dilation[0])
            if isinstance(args.edge_dilation, list) and len(args.edge_dilation) > 0
            else int(args.edge_dilation)
        )
        return {
            "divergence": float(args.divergence),
            "convergence": float(args.convergence),
            "ipd_offset": float(args.ipd_offset),
            "foreground_scale": float(args.foreground_scale),
            "edge_dilation": edge_dilation,
            "ema_normalize": bool(args.ema_normalize),
            "ema_decay": float(args.ema_decay),
            "depth_aa": bool(args.depth_aa),
            "preserve_screen_border": bool(args.preserve_screen_border),
            "synthetic_view": str(args.synthetic_view),
            "stream_fps": int(args.stream_fps),
            "stream_quality": int(args.stream_quality),
            "stereo_format": resolve_player_stereo_format(args),
            "mapper": getattr(args, "mapper", None),
        }

    if args_lock is not None:
        with args_lock:
            return _read()
    return _read()


def read_runtime_adjustment_values(args: Any, lock: Any = None) -> tuple[float, float, float, int, str] | None:
    if args is None:
        return None
    if lock is None:
        state = getattr(args, "state", None)
        lock = state.get("args_lock") if isinstance(state, dict) else None

    def _read() -> tuple[float, float, float, int, str]:
        edge_dilation = args.edge_dilation
        if isinstance(edge_dilation, (list, tuple)) and len(edge_dilation) >= 2:
            edge_x = int(edge_dilation[0])
            edge_y = str(int(edge_dilation[1]))
        elif isinstance(edge_dilation, (list, tuple)) and len(edge_dilation) == 1:
            edge_x = int(edge_dilation[0])
            edge_y = ""
        else:
            edge_x = int(edge_dilation)
            edge_y = ""
        return (
            float(args.divergence),
            float(args.convergence),
            float(args.foreground_scale),
            edge_x,
            edge_y,
        )

    if lock is not None:
        with lock:
            return _read()
    return _read()


def update_runtime_stream_settings(args: Any, updates: dict[str, Any]) -> dict[str, Any]:
    state = getattr(args, "state", None)
    args_lock = state.get("args_lock") if isinstance(state, dict) else None

    def _apply() -> None:
        depth_model = state.get("depth_model") if isinstance(state, dict) else None
        if updates.get("divergence") is not None:
            args.divergence = float(updates["divergence"])
        if updates.get("convergence") is not None:
            args.convergence = float(updates["convergence"])
            convergence_model = state.get("convergence_model") if isinstance(state, dict) else None
            if convergence_model is not None:
                convergence_model.convergence = args.convergence
        if updates.get("ipd_offset") is not None:
            args.ipd_offset = float(updates["ipd_offset"])
        if updates.get("foreground_scale") is not None:
            args.foreground_scale = float(updates["foreground_scale"])
            if isinstance(state, dict):
                state["mapper_overridden"] = True
            if depth_model is not None and depth_model.loaded():
                args.mapper = IW3U.resolve_mapper_name(
                    mapper=None,
                    foreground_scale=args.foreground_scale,
                    metric_depth=depth_model.is_metric(),
                )
        if updates.get("edge_dilation") is not None:
            args.edge_dilation = int(updates["edge_dilation"])
        if updates.get("ema_normalize") is not None:
            args.ema_normalize = bool(updates["ema_normalize"])
        if updates.get("ema_decay") is not None:
            args.ema_decay = float(updates["ema_decay"])
            if depth_model is not None and depth_model.loaded():
                depth_model.enable_ema(args.ema_decay, buffer_size=1)
        if updates.get("depth_aa") is not None:
            args.depth_aa = bool(updates["depth_aa"])
        if updates.get("preserve_screen_border") is not None:
            args.preserve_screen_border = bool(updates["preserve_screen_border"])
        if updates.get("synthetic_view") is not None:
            args.synthetic_view = str(updates["synthetic_view"])
        if updates.get("stream_fps") is not None:
            args.stream_fps = max(1, int(updates["stream_fps"]))
        if updates.get("stream_quality") is not None:
            args.stream_quality = max(1, min(100, int(updates["stream_quality"])))

    if args_lock is not None:
        with args_lock:
            _apply()
    else:
        _apply()
    return get_runtime_stream_settings(args)


def push_stream_frame(frame_sink: Any, sbs: torch.Tensor, args: Any, tick: float) -> None:
    if getattr(args, "local_viewer", False):
        frame_sink.set_frame_data((sbs, tick))
        return
    if args.gpu_jpeg:
        frame_sink.set_frame_data(to_jpeg_data(sbs, quality=args.stream_quality, tick=tick, gpu_jpeg=args.gpu_jpeg))
    else:
        with torch.no_grad():
            sbs_gpu_clone = sbs.detach().clone().contiguous()
        frame_sink.set_frame_data(
            lambda sbs=sbs_gpu_clone, quality=args.stream_quality, t=tick: to_jpeg_data(
                sbs, quality=quality, tick=t, gpu_jpeg=False
            )
        )


def init_desktop_pipeline(
    args: Any,
    init_wxapp: bool = False,
    stop_event: threading.Event | None = None,
) -> SimpleNamespace:
    init_num_threads(args.gpu[0])
    ensure_default_stereo_format(args)
    screenshot_factory = validate_screenshot_args(args)

    device = create_device(args.gpu)
    depth_model = args.state["depth_model"]
    effective_stop_event = stop_event if stop_event is not None else args.state.get("stop_event")
    screenshot_thread = None

    try:
        if args.screenshot == "obs":
            screenshot_thread = screenshot_factory(
                fps=args.stream_fps,
                frame_width=None,
                frame_height=None,
                monitor_index=args.monitor_index,
                window_name=args.window_name,
                device=device,
                crop_top=args.crop_top,
                crop_left=args.crop_left,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
                draw_cursor_enabled=not args.disable_draw_cursor,
            )
            screen_width, screen_height = screenshot_thread.init_capture(stop_event=effective_stop_event)
        elif args.window_name:
            rect = get_window_rect_by_title(args.window_name)
            if rect is None:
                raise RuntimeError(f"window_name={args.window_name} not found")
            screen_width = rect["width"] - args.crop_left - args.crop_right
            screen_height = rect["height"] - args.crop_top - args.crop_bottom
        else:
            size_list = get_monitor_size_list()
            if args.monitor_index >= len(size_list):
                raise RuntimeError(f"monitor_index={args.monitor_index} not found")
            screen_width, screen_height = size_list[args.monitor_index]

        if not depth_model.loaded():
            depth_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)

        # Use Flicker Reduction to prevent 3D sickness
        depth_model.enable_ema(args.ema_decay, buffer_size=1)
        if args.state.get("convergence_model") is not None:
            args.state["convergence_model"].reset(enable_ema=True, decay=0.98)

        initial_mapper = args.state.setdefault("initial_mapper", args.mapper)
        mapper_arg = None if args.state.get("mapper_overridden") else initial_mapper
        args.mapper = IW3U.resolve_mapper_name(
            mapper=mapper_arg, foreground_scale=args.foreground_scale, metric_depth=depth_model.is_metric()
        )

        # TODO: For mlbw, it is better to switch models when the divergence value dynamically changes
        side_model = IW3U.create_stereo_model(
            args.method,
            divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
            device_id=args.gpu[0],
        )

        screen_size = (screen_width, screen_height)
        if screen_height > args.stream_height:
            frame_height = args.stream_height
            frame_width = math.ceil((args.stream_height / screen_height) * screen_width)
            frame_width -= frame_width % 2
        else:
            frame_height = screen_height
            frame_width = screen_width

        if screenshot_thread is not None:
            screenshot_thread.set_frame_size(frame_width, frame_height)

        output_frame_height, output_frame_width = test_output_size(
            (frame_height, frame_width), args, depth_model, side_model
        )

        if init_wxapp and wx.GetApp() is None:
            _empty_app = wx.App()  # needed to initialize wx.GetMousePosition()

        if screenshot_thread is None:
            screenshot_thread = screenshot_factory(
                fps=args.stream_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                monitor_index=args.monitor_index,
                window_name=args.window_name,
                device=device,
                crop_top=args.crop_top,
                crop_left=args.crop_left,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
                draw_cursor_enabled=not args.disable_draw_cursor,
            )

        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model, device=device)
        gc_collect()
    except Exception:
        if screenshot_thread is not None:
            screenshot_thread.stop()
        raise

    return SimpleNamespace(
        device=device,
        depth_model=depth_model,
        side_model=side_model,
        screenshot_thread=screenshot_thread,
        screen_size=screen_size,
        frame_width=frame_width,
        frame_height=frame_height,
        output_frame_width=output_frame_width,
        output_frame_height=output_frame_height,
    )


def run_desktop_processing_loop(
    args: Any,
    pipeline: SimpleNamespace,
    frame_sink: Any,
    stop_event: threading.Event | None = None,
    idle_check: Callable[[], bool] | None = None,
) -> None:
    count = 0
    last_status_time = 0.0
    fps_counter: deque[float] = deque(maxlen=120)
    prev_divergence = args.divergence
    depth_model = pipeline.depth_model
    side_model = pipeline.side_model
    screenshot_thread = pipeline.screenshot_thread
    screen_size = pipeline.screen_size
    frame_width = pipeline.frame_width
    frame_height = pipeline.frame_height

    while True:
        with args.state["args_lock"]:
            tick = time.perf_counter()
            frame = screenshot_thread.get_frame()
            if frame is None:
                break
            sbs = IW3U.process_image(frame, args, depth_model, side_model, autocrop_uncrop=True)
            push_stream_frame(frame_sink, sbs, args, tick)

            if count > 1 and tick - last_status_time > 1:
                last_status_time = tick
                mean_processing_time = sum(fps_counter) / len(fps_counter)
                estimated_fps = 1.0 / mean_processing_time
                screen_size_tuple = (screen_size, (frame_width, frame_height))
                if args.state["fps_event"] is not None:
                    args.state["fps_event"].update(
                        estimated_fps, screenshot_thread.get_fps(), frame_sink.get_fps(), screen_size_tuple
                    )
                else:
                    print(
                        f"\rEstimated FPS = {estimated_fps:.02f}, "
                        f"Screenshot FPS = {screenshot_thread.get_fps():.02f}, "
                        f"Streaming FPS = {frame_sink.get_fps():.02f}, "
                        f"Screen Size = {screen_size_tuple}",
                        end="",
                    )

        if getattr(args, "local_viewer", False):
            # Platform-specific behavior for local viewer
            if sys.platform != "win32":
                # Linux: Use wx.Yield() (original working behavior)
                if not wx.GetApp().IsMainLoopRunning():
                    wx.Yield()
            if frame_sink.is_closed():
                break

        if prev_divergence != args.divergence:
            if args.method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
                side_model = try_switch_mlbw_model(
                    old_divergence=prev_divergence,
                    new_divergence=args.divergence,
                    old_side_model=side_model,
                    args=args,
                )
            prev_divergence = args.divergence

        process_time = time.perf_counter() - tick
        wait_time = max((1 / args.stream_fps) - process_time, 0)
        time.sleep(wait_time)
        fps_counter.append(process_time)
        count += 1
        if stop_event is not None and stop_event.is_set():
            break
        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
            break
        if idle_check is not None and idle_check():
            break


def cleanup_desktop_pipeline(pipeline: SimpleNamespace | None, depth_model: Any) -> None:
    if pipeline is not None and pipeline.screenshot_thread is not None:
        pipeline.screenshot_thread.stop()
    depth_model.clear_compiled_model()
    depth_model.reset()
    gc_collect()


def iw3_desktop_main(args, init_wxapp=True):
    ensure_default_stereo_format(args)

    if not args.local_viewer:
        auth = validate_bind_address_and_auth(args)
    else:
        auth = None

    validate_screenshot_args(args)

    depth_model = args.state["depth_model"]
    pipeline = None
    server = None
    try:
        pipeline = init_desktop_pipeline(args, init_wxapp=init_wxapp)
        lock = threading.Lock()

        if not args.local_viewer:
            # Web Streaming
            if getattr(args, "legacy_stream", False):
                index_tpl_path = path.join(path.dirname(__file__), "views", "index.html.tpl")
                with open(index_tpl_path, mode="r", encoding="utf-8") as f:
                    index_template = f.read()

                server = StreamingServer(
                    host=args.bind_addr,
                    port=args.port,
                    lock=lock,
                    frame_width=pipeline.output_frame_width,
                    frame_height=pipeline.output_frame_height,
                    fps=args.stream_fps,
                    index_template=index_template,
                    stream_uri="/stream.jpg",
                    stream_content_type="image/jpeg",
                    auth=auth,
                )
            else:
                from ..player.download_assets import main as download_player_assets

                download_player_assets()
                server = WebXRStreamingServer(
                    host=args.bind_addr,
                    port=args.port,
                    lock=lock,
                    frame_width=pipeline.output_frame_width,
                    frame_height=pipeline.output_frame_height,
                    fps=args.stream_fps,
                    stereo_format=resolve_player_stereo_format(args),
                    stream_uri="/stream.jpg",
                    stream_content_type="image/jpeg",
                    auth=auth,
                    args=args,
                )
        else:
            # Local Viewer
            if LocalViewer is None:
                raise RuntimeError("Local Viewer is not available")
            IS_ROCM = getattr(torch.version, "hip", None) is not None
            USE_CUDA = torch.cuda.is_available() and not IS_ROCM
            server = LocalViewer(
                lock=lock,
                width=pipeline.output_frame_width,
                height=pipeline.output_frame_height,
                use_cuda=USE_CUDA,
                uncap_fps=args.uncap_fps,
                fit_window=args.fit_window,
            )

        # main loop
        server.start()
        pipeline.screenshot_thread.start()

        if not args.local_viewer:
            scheme = "http" if getattr(args, "legacy_stream", False) else "https"
            stream_url = f"{scheme}://{args.bind_addr}:{args.port}"
            if args.state["fps_event"] is not None:
                args.state["fps_event"].set_url(stream_url)
            else:
                print(f"Open {stream_url}")
        else:
            if args.state["fps_event"] is not None:
                args.state["fps_event"].set_url(None)

        run_desktop_processing_loop(args, pipeline, server)
    finally:
        if server is not None:
            server.stop()
        cleanup_desktop_pipeline(pipeline, depth_model)

    if args.state["stop_event"] and args.state["stop_event"].is_set():
        args.state["stop_event"].clear()

    return args
