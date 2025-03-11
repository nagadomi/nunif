import threading
import math
from os import path
import io
import time
from collections import deque
import wx  # for mouse pointer
import torch
from torchvision.io import encode_jpeg
from .. import models # noqa
from ..utils import (
    create_parser, set_state_args, process_image,
    ROW_FLOW_V2_URL, ROW_FLOW_V3_URL, ROW_FLOW_V3_SYM_URL,
    HUB_MODEL_DIR, resolve_mapper_name
)
import socket
import struct
from nunif.utils.ui import TorchHubDir
from nunif.device import create_device
from nunif.initializer import gc_collect
from nunif.models import load_model
from .streaming_server import StreamingServer
from .screenshot_thread_pil import ScreenshotThreadPIL, take_screenshot
from .screenshot_process import ScreenshotProcess


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


def load_side_model(args):
    with TorchHubDir(HUB_MODEL_DIR):
        if args.method in {"row_flow_v3", "row_flow"}:
            side_model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = False
            side_model.delta_output = True
        elif args.method in {"row_flow_v3_sym", "row_flow_sym"}:
            side_model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = True
            side_model.delta_output = True
        elif args.method == "row_flow_v2":
            side_model = load_model(ROW_FLOW_V2_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.delta_output = True
        else:
            side_model = None

    return side_model


def to_uint8_cpu(x):
    return x.mul(255).round_().to(torch.uint8).cpu()


def to_jpeg_data(frame, quality, tick):
    bio = io.BytesIO()
    frame = to_uint8_cpu(frame)
    # TODO: encode_jpeg has a bug with cuda, but that will be fixed in the next version.
    frame = encode_jpeg(frame, quality=quality)
    bio.write(frame.numpy())
    return (bio.getbuffer().tobytes(), tick)


def main():
    local_address = get_local_address()
    parser = create_parser(required_true=False)
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
    args = parser.parse_args()

    if not args.full_sbs:
        args.half_sbs = True
        frame_width_scale = 1
    else:
        frame_width_scale = 2

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

    # initialize
    set_state_args(args)
    device = create_device(args.gpu)

    args.bg_session = None
    if args.edge_dilation is None:
        args.edge_dilation = 2

    depth_model = args.state["depth_model"]
    if not depth_model.loaded():
        depth_model.load(gpu=args.gpu, resolution=args.resolution)
    # Use Flicker Reduction to prevent 3D sickness
    depth_model.enable_ema_minmax(args.ema_decay)

    args.mapper = resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                      metric_depth=depth_model.is_metric())
    side_model = load_side_model(args)
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

    # main loop
    server.start()
    screenshot_thread.start()
    print(f"Open http://{args.bind_addr}:{args.port}")
    count = 0
    fps_counter = deque(maxlen=120)
    try:
        while True:
            tick = time.time()
            frame = screenshot_thread.get_frame()
            sbs = process_image(frame, args, depth_model, side_model, return_tensor=True)
            server.set_frame_data(lambda: to_jpeg_data(sbs, quality=args.stream_quality, tick=tick))
            count += 1
            if count % 300 == 0:
                gc_collect()

            process_time = time.time() - tick
            wait_time = max((1 / (args.stream_fps)) - process_time, 0)
            if count > 1:
                fps_counter.append(process_time)
                mean_processing_time = sum(fps_counter) / len(fps_counter)
                estimated_fps = 1.0 / mean_processing_time
                if count % 4 == 0:
                    print(f"\rEstimated FPS = {estimated_fps:.02f}, "
                          f"Screenshot FPS = {screenshot_thread.get_fps():.02f}, "
                          f"Streaming FPS = {server.get_fps():.02f}", end="")
            time.sleep(wait_time)
    finally:
        server.stop()
        screenshot_thread.stop()


if __name__ == "__main__":
    main()
