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
from packaging.version import Version
import torch
from torchvision.io import encode_jpeg
from .. import utils as IW3U
from .. import models  # noqa
from .screenshot_thread_pil import ScreenshotThreadPIL
from .screenshot_process import ( # noqa
    ScreenshotProcess,
    get_monitor_size_list,
    get_window_rect_by_title,
    enum_window_names,
)
from .streaming_server import StreamingServer
from nunif.device import create_device
from nunif.models import compile_model
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.initializer import gc_collect


TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)
TORCH_NUM_THREADS = torch.get_num_threads()


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


def init_num_threads(device_id):
    if device_id < 0:
        # cpu
        torch.set_num_threads(TORCH_NUM_THREADS)
    else:
        torch.set_num_threads(1)


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


def to_uint8(x):
    return x.mul(255).round_().to(torch.uint8)


def fps_sleep(start_time, fps, resolution=2e-4):
    # currently not in used
    end_diff_time = (1 / fps) - resolution
    while time.perf_counter() - start_time < end_diff_time:
        time.sleep(resolution)


def to_jpeg_data(frame, quality, tick, gpu_jpeg=True):
    bio = io.BytesIO()
    if ENABLE_GPU_JPEG and gpu_jpeg and frame.device.type == "cuda":
        jpeg_data = encode_jpeg(to_uint8(frame), quality=quality).cpu()
    else:
        jpeg_data = encode_jpeg(to_uint8(frame).cpu(), quality=quality)
    bio.write(jpeg_data.numpy())
    jpeg_data = bio.getbuffer().tobytes()
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
    parser.add_argument("--gpu-jpeg", action="store_true", help="Use GPU JPEG Encoder")
    parser.add_argument("--monitor-index", type=int, default=0, help="monitor_index for wc_mp. 0 origin. 0 = monitor 1")
    parser.add_argument("--window-name", type=str, help=("target window name for wc_mp."
                                                         " When this is specified, --monitor-index is ignored"))
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
    init_num_threads(args.gpu[0])

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

    if args.screenshot != "wc_mp" and args.monitor_index != 0:
        raise RuntimeError(f"{args.screenshot} does not support monitor_index={args.monitor_index}")
    if args.screenshot != "wc_mp" and args.window_name:
        raise RuntimeError(f"{args.screenshot} does not support --window-name option")

    if args.window_name:
        rect = get_window_rect_by_title(args.window_name)
        if rect is None:
            raise RuntimeError(f"window_name={args.window_name} not found")
        screen_width, screen_height = rect["width"], rect["height"]
    else:
        size_list = get_monitor_size_list()
        if args.monitor_index >= len(size_list):
            raise RuntimeError(f"monitor_index={args.monitor_index} not found")
        screen_width, screen_height = size_list[args.monitor_index]

    screen_size = (screen_width, screen_height)
    if screen_height > args.stream_height:
        frame_height = args.stream_height
        frame_width = math.ceil((args.stream_height / screen_height) * screen_width)
        frame_width -= frame_width % 2
    else:
        frame_height = screen_height
        frame_width = screen_width

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
        monitor_index=args.monitor_index, window_name=args.window_name,
        device=device)

    try:
        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model)
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
                if args.gpu_jpeg:
                    server.set_frame_data(to_jpeg_data(sbs, quality=args.stream_quality, tick=tick, gpu_jpeg=args.gpu_jpeg))
                else:
                    server.set_frame_data(lambda: to_jpeg_data(sbs, quality=args.stream_quality, tick=tick, gpu_jpeg=args.gpu_jpeg))

                if count % (args.stream_fps * 30) == 0:
                    gc_collect()
                if count > 1 and count % args.stream_fps == 0:
                    mean_processing_time = sum(fps_counter) / len(fps_counter)
                    estimated_fps = 1.0 / mean_processing_time
                    screen_size_tuple = (screen_size, (frame_width, frame_height))
                    if args.state["fps_event"] is not None:
                        args.state["fps_event"].update(estimated_fps, screenshot_thread.get_fps(),
                                                       server.get_fps(), screen_size_tuple)
                    else:
                        print(f"\rEstimated FPS = {estimated_fps:.02f}, "
                              f"Screenshot FPS = {screenshot_thread.get_fps():.02f}, "
                              f"Streaming FPS = {server.get_fps():.02f}, "
                              f"Screen Size = {screen_size_tuple}", end="")

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
        depth_model.clear_compiled_model()

    if args.state["stop_event"] and args.state["stop_event"].is_set():
        args.state["stop_event"].clear()

    return args
