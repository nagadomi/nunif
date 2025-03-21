import threading
import io
import sys
import math
from os import path
import time
import socket
import struct
from collections import deque
import wx  # for mouse pointer
import torch
from torchvision.io import encode_jpeg
from .. import utils as IW3U
from .. import models  # noqa
from .screenshot_thread_pil import ScreenshotThreadPIL, take_screenshot
from .screenshot_process import ScreenshotProcess
from .streaming_server import StreamingServer
from nunif.device import create_device
from nunif.initializer import gc_collect


def init_win32():
    if sys.platform == "win32":
        import ctypes
        try:
            # Fix mouse position when Display Scaling is not 100%
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except: # noqa
            pass

        # if sys.version_info <= (3, 11):  # python 3.11 or later has high precision sleep.
        try:
            # Change timer/sleep precision
            ctypes.windll.winmm.timeBeginPeriod(1)
        except: # noqa
            pass


def get_local_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 8.8.8.8 is Google DNS address
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: # noqa
        return "127.0.0.1"  # unknown


def is_private_address(ip):
    """ https://stackoverflow.com/a/8339939
    """
    f = struct.unpack("!I", socket.inet_pton(socket.AF_INET, ip))[0]
    private = (
        [2130706432, 4278190080],  # 127.0.0.0,   255.0.0.0   https://www.rfc-editor.org/rfc/rfc3330
        [3232235520, 4294901760],  # 192.168.0.0, 255.255.0.0 https://www.rfc-editor.org/rfc/rfc1918
        [2886729728, 4293918720],  # 172.16.0.0,  255.240.0.0 https://www.rfc-editor.org/rfc/rfc1918
        [167772160, 4278190080],   # 10.0.0.0,    255.0.0.0   https://www.rfc-editor.org/rfc/rfc1918
    )
    for net in private:
        if (f & net[1]) == net[0]:
            return True
    return False


def to_uint8_cpu(x):
    return x.mul(255).round_().to(torch.uint8).cpu()


def fps_sleep(start_time, fps, resolution=2e-4):
    # currently not in used
    end_diff_time = (1 / fps) - resolution
    while time.perf_counter() - start_time < end_diff_time:
        time.sleep(resolution)


def to_jpeg_data(frame, quality, tick):
    bio = io.BytesIO()
    frame = to_uint8_cpu(frame)
    # TODO: encode_jpeg has a bug with cuda, but that will be fixed in the next version.
    frame = encode_jpeg(frame, quality=quality)
    bio.write(frame.numpy())
    return (bio.getbuffer().tobytes(), tick)


def create_parser():
    local_address = get_local_address()
    parser = IW3U.create_parser(required_true=False)
    parser.add_argument("--port", type=int, default=1303,
                        help="HTTP listen port")
    parser.add_argument("--bind-addr", type=str, default=local_address,
                        help="HTTP listen address")
    parser.add_argument("--user", type=str, help="HTTP Basic Authentication username")
    parser.add_argument("--password", type=str, help="HTTP Basic Authentication password")
    parser.add_argument("--stream-fps", type=int, default=30, help="Streaming FPS")
    parser.add_argument("--stream-height", type=int, default=1080, help="Streaming screen resolution")
    parser.add_argument("--stream-quality", type=int, default=90, help="Streaming JPEG quality")
    parser.add_argument("--full-sbs", action="store_true", help="Use Full SBS for Pico4")
    parser.add_argument("--screenshot", type=str, default="pil", choices=["pil", "pil_mp", "wc_mp"],
                        help="Screenshot method")
    parser.set_defaults(
        input="dummy",
        output="dummy",
        depth_model="Any_V2_S",
        divergence=1.0,
        convergence=1.0,
        ema_normalize=True,
    )
    return parser


def set_state_args(args, args_lock=None, stop_event=None, fps_event=None, depth_model=None):
    IW3U.set_state_args(args, stop_event=stop_event, depth_model=depth_model)
    args.state["fps_event"] = fps_event
    args.state["args_lock"] = args_lock if args_lock is not None else threading.Lock()
    args.bg_session = None
    if args.edge_dilation is None:
        args.edge_dilation = 2


def iw3_desktop_main(args, init_wxapp=True):
    if not args.full_sbs:
        args.half_sbs = True
        frame_width_scale = 1
    else:
        frame_width_scale = 2

    if args.bind_addr is None:
        args.bind_addr = get_local_address()
    if args.bind_addr == "0.0.0.0":
        pass  # Allows specifying undefined addresses
    elif args.bind_addr == "127.0.0.1" or not is_private_address(args.bind_addr):
        raise RuntimeError(f"Detected IP address({args.bind_addr}) is not Local Area Network Address."
                           " Specify --bind-addr option")

    if args.screenshot == "pil":
        screenshot_factory = ScreenshotThreadPIL
    elif args.screenshot == "pil_mp":
        screenshot_factory = lambda *args, **kwargs: ScreenshotProcess(*args, **kwargs, backend="pil")
    elif args.screenshot == "wc_mp":
        screenshot_factory = lambda *args, **kwargs: ScreenshotProcess(*args, **kwargs, backend="windows_capture")

    device = create_device(args.gpu)

    depth_model = args.state["depth_model"]
    if not depth_model.loaded():
        depth_model.load(gpu=args.gpu, resolution=args.resolution)
    # Use Flicker Reduction to prevent 3D sickness
    depth_model.enable_ema_minmax(args.ema_decay)
    args.mapper = IW3U.resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                           metric_depth=depth_model.is_metric())
    side_model = IW3U.load_sbs_model(args)
    if args.user or args.password:
        user = args.user or ""
        password = args.password or ""
        auth = (user, password)
    else:
        auth = None

    frame = take_screenshot()
    if frame.height > args.stream_height:
        frame_height = args.stream_height
        frame_width = math.ceil((args.stream_height / frame.height) * frame.width)
        frame_width -= frame_width % 2
    else:
        frame_height = frame.height
        frame_width = frame.width

    with open(path.join(path.dirname(__file__), "views", "index.html.tpl"),
              mode="r", encoding="utf-8") as f:
        index_template = f.read()

    if init_wxapp:
        empty_app = wx.App()  # noqa: this is needed to initialize wx.GetMousePosition()

    lock = threading.Lock()
    server = StreamingServer(
        host=args.bind_addr,
        port=args.port, lock=lock,
        frame_width=frame_width * frame_width_scale,
        frame_height=frame_height,
        fps=args.stream_fps,
        index_template=index_template,
        stream_uri="/stream.jpg", stream_content_type="image/jpeg",
        auth=auth
    )
    screenshot_thread = screenshot_factory(
        fps=args.stream_fps,
        frame_width=frame_width, frame_height=frame_height,
        device=device)

    try:
        # main loop
        server.start()
        screenshot_thread.start()
        if args.state["fps_event"] is not None:
            args.state["fps_event"].set_url(f"http://{args.bind_addr}:{args.port}")
        else:
            print(f"Open http://{args.bind_addr}:{args.port}")
        count = 0
        fps_counter = deque(maxlen=120)

        while True:
            with args.state["args_lock"]:
                tick = time.perf_counter()
                frame = screenshot_thread.get_frame()
                sbs = IW3U.process_image(frame, args, depth_model, side_model, return_tensor=True)
                server.set_frame_data(lambda: to_jpeg_data(sbs, quality=args.stream_quality, tick=tick))

                if count % (args.stream_fps * 30) == 0:
                    gc_collect()
                if count > 1 and count % args.stream_fps == 0:
                    mean_processing_time = sum(fps_counter) / len(fps_counter)
                    estimated_fps = 1.0 / mean_processing_time
                    if args.state["fps_event"] is not None:
                        args.state["fps_event"].update(estimated_fps, screenshot_thread.get_fps(), server.get_fps())
                    else:
                        print(f"\rEstimated FPS = {estimated_fps:.02f}, "
                              f"Screenshot FPS = {screenshot_thread.get_fps():.02f}, "
                              f"Streaming FPS = {server.get_fps():.02f}", end="")

            process_time = time.perf_counter() - tick
            wait_time = max((1 / (args.stream_fps)) - process_time, 0)
            time.sleep(wait_time)
            fps_counter.append(process_time)
            count += 1
            if args.state["stop_event"] and args.state["stop_event"].is_set():
                break
    finally:
        server.stop()
        screenshot_thread.stop()

    if args.state["stop_event"] and args.state["stop_event"].is_set():
        args.state["stop_event"].clear()

    return args
