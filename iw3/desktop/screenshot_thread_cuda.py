import threading
import torch
import torch.nn.functional as F
from collections import deque
import time


def resize_frame(frame, size):
    # CHW uint8 - > CHW float
    frame = frame.unsqueeze(0).float().div_(255.0)
    return F.interpolate(
        frame,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


class ScreenshotThreadWCCUDA(threading.Thread):
    def __init__(self, fps, frame_width, frame_height, monitor_index, window_name, device,
                 crop_top=0, crop_left=0, crop_right=0, crop_bottom=0,
                 **_ignore_unsupported_kwargs):
        super().__init__(daemon=True)
        if device.type != "cuda":
            raise RuntimeError("wc_cuda requires a CUDA device")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.device = device
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.frame = None
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        self.tick = 0

    def prepare_frame(self, source_frame):
        output_size = (self.frame_height, self.frame_width)
        if source_frame.shape[-2:] != output_size:
            if self.window_name is not None:
                # Window capture can fluctuate by a few pixels due to window effects or resizing.
                frame = torch.zeros((3, *output_size), device=self.device, dtype=torch.float32)
                min_h = min(frame.shape[1], source_frame.shape[1])
                min_w = min(frame.shape[2], source_frame.shape[2])
                frame[:, 0:min_h, 0:min_w].copy_(source_frame[:, 0:min_h, 0:min_w])
                frame.div_(255.0)
            else:
                frame = resize_frame(source_frame, size=output_size)
        else:
            frame = source_frame.float().div_(255.0)
        return frame

    def run(self):
        from wc_cuda import WindowsCapture
        if self.window_name:
            # ignore
            monitor_index = None
        else:
            # 1 origin
            monitor_index = self.monitor_index + 1

        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=monitor_index,
            window_name=self.window_name,
            device_id=self.device.index,
        )

        @capture.event
        def on_frame_arrived(frame, capture_control):
            # BGRA HWC -> RGB CHW
            source_frame = frame.frame_buffer[..., [2, 1, 0]].permute(2, 0, 1)
            if self.window_name and (self.crop_top > 0 or self.crop_left > 0 or self.crop_right > 0 or self.crop_bottom > 0):
                h, w = source_frame.shape[-2:]
                top = self.crop_top
                bottom = h - self.crop_bottom if self.crop_bottom > 0 else h
                left = self.crop_left
                right = w - self.crop_right if self.crop_right > 0 else w
                source_frame = source_frame[:, top:bottom, left:right]

            prepared_frame = self.prepare_frame(source_frame)

            with self.frame_lock:
                self.frame = prepared_frame
                self.frame_set_event.set()

            now = time.perf_counter()
            if self.tick > 0:
                process_time = now - self.tick
                with self.fps_lock:
                    self.fps_counter.append(process_time)
            self.tick = now

            if self.stop_event.is_set():
                capture_control.stop()

        @capture.event
        def on_closed():
            pass

        try:
            # event loop
            capture.start()
        finally:
            self.frame_set_event.set()
            time.sleep(0.1)

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if not self.is_alive():
                raise RuntimeError("thread is already dead")
        with self.frame_lock:
            frame = self.frame
            self.frame = None
            self.frame_set_event.clear()

        if frame is None:
            raise RuntimeError("thread is dead")

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
            self.join(timeout=4)
