from PIL import ImageGrab, ImageDraw
import numpy as np
import threading
import torch
from collections import deque
import time
import wx
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)


def take_screenshot(mouse_position=None):
    frame = ImageGrab.grab(include_layered_windows=True)
    if frame.mode != "RGB":
        frame = frame.convert("RGB")
    if mouse_position is not None:
        gc = ImageDraw.Draw(frame)
        gc.circle(mouse_position, radius=4, fill=None, outline=(0x33, 0x80, 0x80), width=2)

    return frame


def to_tensor(pil_image, device, frame_buffer):
    # Transfer the image data to VRAM as uint8 first, then convert it to float.
    x = np.array(pil_image)
    x = frame_buffer.copy_(torch.from_numpy(x).permute(2, 0, 1)).to(device)
    x = x / 255.0  # to float
    return x


class ScreenshotThreadPIL(threading.Thread):
    def __init__(self, fps, frame_width, frame_height, monitor_index, window_name, device, **_ignore_unsupported_kwargs):
        super().__init__(daemon=True)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index  # TODO: not implemented
        self.window_name = window_name  # TODO: not implemented
        self.device = device
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.frame = None
        self.frame_unset_event = threading.Event()
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        if device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=device)
        else:
            self.cuda_stream = None

    def run(self):
        frame_buffer = None
        while True:
            tick = time.perf_counter()
            frame = take_screenshot(wx.GetMousePosition())
            if frame_buffer is None:
                frame_buffer = torch.ones((3, frame.height, frame.width), dtype=torch.uint8)
                if torch.cuda.is_available():
                    frame_buffer = frame_buffer.pin_memory()
            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    frame = to_tensor(frame, self.device, frame_buffer)
                    if frame.shape[2] > self.frame_height:
                        frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                          interpolation=InterpolationMode.BILINEAR,
                                          antialias=True)
                    self.cuda_stream.synchronize()
            else:
                frame = to_tensor(frame, self.device, frame_buffer)
                if frame.shape[2] > self.frame_height:
                    frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                      interpolation=InterpolationMode.BILINEAR,
                                      antialias=True)

            with self.frame_lock:
                self.frame = frame
                self.frame_set_event.set()
                self.frame_unset_event.clear()

            process_time = time.perf_counter() - tick
            with self.fps_lock:
                self.fps_counter.append(process_time)

            if self.stop_event.is_set():
                break
            self.frame_unset_event.wait()

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if not self.is_alive():
                raise RuntimeError("thread is already dead")
        with self.frame_lock:
            frame = self.frame
            self.frame = None
            self.frame_set_event.clear()
            self.frame_unset_event.set()
        assert frame is not None
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
        self.frame_unset_event.set()
        if self.ident is not None:
            self.join(timeout=4)
