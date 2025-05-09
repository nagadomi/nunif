import threading
import torch
from collections import deque
import time
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import ctypes
import sys
import wx
from PIL import ImageGrab


class FramePIL():
    def __init__(self, frame):
        self.frame_buffer = frame


class CaptureControlPIL():
    def __init__(self):
        self._stop = False

    def stop(self):
        self._stop = True


class WindowsCapturePIL():
    def __init__(self, *args, **kwargs):
        pass

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.on_frame_arrived = handler
        elif handler.__name__ == "on_closed":
            self.on_closed = handler
        else:
            raise ValueError(handler.__name__)

    def start(self):
        control = CaptureControlPIL()
        frame_buffer = None
        while True:
            tick = time.perf_counter()
            frame = ImageGrab.grab()
            if frame.mode != "RGB":
                frame.convert("RGB")
            rgb = np.array(frame)
            if frame_buffer is None:
                frame_buffer = np.ones((rgb.shape[0], rgb.shape[1], 4), dtype=rgb.dtype)
            # to BGRA
            frame_buffer[:, :, 0:3] = rgb[:, :, ::-1]
            self.on_frame_arrived(FramePIL(frame_buffer), control)
            if control._stop:
                self.on_closed()
                break

            process_time = time.perf_counter() - tick
            wait_time = max((1 / 60) - process_time, 0)
            time.sleep(wait_time)


def draw_cursor(x, pos, size=8):
    C, H, W = x.shape
    r = size // 2
    rr = r // 2
    pos_x = min(max(pos[0], r), W - r)
    pos_y = min(max(pos[1], r), H - r)
    px = x[:, pos_y - rr: pos_y + rr, pos_x - rr: pos_x + rr].clone()
    color = torch.tensor((0x33 / 255.0, 0x80 / 255.0, 0x80 / 255.0), dtype=px.dtype, device=px.device).view(3, 1, 1)
    x[:, pos_y - r: pos_y + r, pos_x - r: pos_x + r] = color
    x[:, pos_y - rr: pos_y + rr, pos_x - rr: pos_x + rr] = px


def get_screen_size():
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    else:
        frame = ImageGrab.grab()
        return frame.size


def estimate_fps(fps_counter):
    diff = []
    prev = None
    for t in fps_counter:
        if prev is not None:
            diff.append(t - prev)
        prev = t
    if diff:
        mean_time = sum(diff) / len(diff)
        return 1 / mean_time
    else:
        return 0


def capture_process(frame_size, frame_shm, frame_lock, frame_event, stop_event, backend="pil"):
    frame_buffer = np.ndarray(frame_size, dtype=np.uint8, buffer=frame_shm.buf)

    if backend == "pil":
        capture = WindowsCapturePIL()
    elif backend == "windows_capture":
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            frame_event.set()
            raise

        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=None,
            window_name=None,
        )

    @capture.event
    def on_frame_arrived(frame, capture_control):
        nonlocal frame_shm, frame_event, frame_lock, stop_event, frame_buffer  # noqa
        if not frame_event.is_set():
            with frame_lock:
                if frame_buffer.shape != frame.frame_buffer.shape:
                    raise RuntimeError("Screen size missmatch")
                frame_buffer[:] = frame.frame_buffer
                frame_event.set()

        if stop_event.is_set():
            capture_control.stop()

    @capture.event
    def on_closed():
        pass

    try:
        # event loop
        capture.start()
    finally:
        frame_event.set()


def to_tensor(bgra, device):
    x = torch.from_numpy(bgra)
    x = x[:, :, :3].permute(2, 0, 1).contiguous().to(device)
    x = x / 255.0
    return x


class ScreenshotProcess(threading.Thread):
    def __init__(self, fps, frame_width, frame_height, device, backend="pil"):
        super().__init__()
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.device = device
        self.frame = None
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        if device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=device)
        else:
            self.cuda_stream = None

    def start_process(self):
        screen_size = get_screen_size()
        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]
        template = np.zeros((self.screen_height, self.screen_width, 4), dtype=np.uint8)
        self.process_frame_buffer = shared_memory.SharedMemory(create=True, size=template.nbytes)
        self.process_stop_event = mp.Event()
        self.process_frame_event = mp.Event()
        self.process_frame_lock = mp.Lock()
        self.process = mp.Process(
            target=capture_process,
            args=(tuple(template.shape),
                  self.process_frame_buffer,
                  self.process_frame_lock,
                  self.process_frame_event,
                  self.process_stop_event,
                  self.backend))
        self.process.start()

    def run(self):
        self.start_process()
        frame_buffer = None
        try:
            while True:
                tick = time.perf_counter()
                self.process_frame_event.clear()
                while not self.process_frame_event.wait(1):
                    if not self.process.is_alive():
                        raise RuntimeError("thread is already dead")
                with self.process_frame_lock:
                    frame = np.ndarray((self.screen_height, self.screen_width, 4),
                                       dtype=np.uint8, buffer=self.process_frame_buffer.buf)
                    # deepcopy
                    frame = torch.from_numpy(frame)
                    if frame_buffer is None:
                        frame_buffer = frame.clone()
                        if torch.cuda.is_available():
                            frame_buffer = frame_buffer.pin_memory()
                    else:
                        frame_buffer.copy_(frame)

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        if self.backend == "pil":
                            # cursor for PIL
                            draw_cursor(frame, wx.GetMousePosition())
                        if frame.shape[1:] != (self.frame_height, self.frame_width):
                            frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                              interpolation=InterpolationMode.BILINEAR,
                                              antialias=True)
                        self.cuda_stream.synchronize()
                else:
                    frame = frame_buffer.to(self.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    if self.backend == "pil":
                        draw_cursor(frame, wx.GetMousePosition())
                    if frame.shape[1:] != (self.frame_height, self.frame_width):
                        frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                          interpolation=InterpolationMode.BILINEAR,
                                          antialias=True)

                with self.frame_lock:
                    self.frame = frame

                process_time = time.perf_counter() - tick
                with self.fps_lock:
                    self.fps_counter.append(process_time)

                if self.stop_event.is_set():
                    break
        finally:
            self.process_stop_event.set()
            self.process.join()
            self.stop_event.set()
            self.process = None

    def get_frame(self):
        frame = None
        while frame is None:
            if self.stop_event.is_set():
                raise RuntimeError("thread is already dead")

            with self.frame_lock:
                frame = self.frame
        return frame

    def get_fps(self):
        with self.fps_lock:
            if self.fps_counter:
                mean_processing_time = sum(self.fps_counter) / len(self.fps_counter)
                return 1 / mean_processing_time
            else:
                return 0

    def stop(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join()
