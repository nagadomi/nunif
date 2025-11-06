import os
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
import sys
import wx
import mss


_mss = mss.mss()


_x11_connection_pool = {}


def get_x11root():
    key = os.getpid()
    root = _x11_connection_pool.get(key)
    if root is None:
        from Xlib import display
        root = display.Display().screen().root
        _x11_connection_pool[key] = root

    return root


class FrameMSS():
    def __init__(self, frame):
        self.frame_buffer = frame


class CaptureControlMSS():
    def __init__(self):
        self._stop = False

    def stop(self):
        self._stop = True


class WindowsCaptureMSS():
    def __init__(self, monitor_index=0, window_name=None):
        self._mss = mss.mss()
        self.window_name = window_name
        self.monitor_index = monitor_index

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.on_frame_arrived = handler
        elif handler.__name__ == "on_closed":
            self.on_closed = handler
        else:
            raise ValueError(handler.__name__)

    def start(self):
        control = CaptureControlMSS()
        while True:
            tick = time.perf_counter()
            if self.window_name is not None:
                position = get_window_rect_by_title(self.window_name)
            else:
                position = self._mss.monitors[self.monitor_index + 1]
            shot = self._mss.grab(position)
            self.on_frame_arrived(FrameMSS(np.asarray(shot)), control)
            if control._stop:
                self.on_closed()
                break

            process_time = time.perf_counter() - tick
            wait_time = max((1 / 60) - process_time, 0)
            time.sleep(wait_time)


def draw_cursor(x, pos, offset={"left": 0, "top": 0}, size=12):
    C, H, W = x.shape
    r = size // 2
    rr = r // 2
    pos_x = min(max(pos[0] - offset["left"], r), W - r)
    pos_y = min(max(pos[1] - offset["top"], r), H - r)
    px = x[:, pos_y - rr: pos_y + rr, pos_x - rr: pos_x + rr].clone()
    color = torch.tensor((0x33 / 255.0, 0x80 / 255.0, 0x80 / 255.0), dtype=px.dtype, device=px.device).view(3, 1, 1)
    x[:, pos_y - r: pos_y + r, pos_x - r: pos_x + r] = color
    x[:, pos_y - rr: pos_y + rr, pos_x - rr: pos_x + rr] = px


def get_monitor_size_list():
    if sys.platform == "win32":
        import win32api
        monitors = win32api.EnumDisplayMonitors()
        size_list = []
        for monitor in monitors:
            sx, sy, width, height = monitor[2]
            width = width - sx
            height = height - sy
            size_list.append((width, height))
        return size_list
    else:  # This doesn't use any platform specific call (safe for all OS)
        monitors = _mss.monitors
        size_list = []
        for monitor in monitors:
            size_list.append((monitor['width'], monitor['height']))
        return size_list[1:]


DENY_WINDOW_NAMES = {
    "Microsoft Text Input Application",
    "Program Manager"
}


def enum_window_names():
    if sys.platform == "win32":
        import win32gui

        window_names = []

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and title not in DENY_WINDOW_NAMES:
                    window_names.append(title)

        win32gui.EnumWindows(callback, None)
        return sorted(window_names)
    elif sys.platform == "linux":
        def _enum_window_names(window=None, root=None):
            if window is None:
                window = root
            try:
                name = window.get_wm_name()
                geom = window.get_geometry()
                abs_pos = root.translate_coords(window, 0, 0)
                # this also saves initial window coordinates and address for uniqueness
                window_names = [str(name) + "|" + str(abs_pos.x) + "," + str(abs_pos.y) + "|" +
                                str(geom.width) + "," + str(geom.height) + "|" + str(window)[-9: -1]]
                if geom.width < 128 or geom.height < 128:
                    window_names = []  # Reject window size smaller than 128 x 128
            except:  # noqa
                window_names = []
            try:
                for child in window.query_tree().children:
                    window_names += _enum_window_names(child, root)
            except:  # noqa
                pass

            return window_names

        return sorted(_enum_window_names(window=None, root=get_x11root()))
    else:
        return []  # WARNING: mac is unimplemented!


def XFindWindow(window, address, root):
    if window is None:
        window = root
    try:
        if str(window)[-9: -1] == address:
            return window
        for child in window.query_tree().children:
            ret = XFindWindow(child, address, root)
            if ret is not None:
                return ret
    except:  # noqa
        pass

    return None


def get_window_rect_by_title(title):
    if sys.platform == "win32":
        import win32gui

        hwnd = win32gui.FindWindow(None, title)
        if hwnd == 0:
            return None

        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }
    elif sys.platform == "linux":
        root = get_x11root()
        comp = title.split("|")
        if len(comp) < 4:
            return None

        window = XFindWindow(None, comp[-1], root)
        try:
            geom = window.get_geometry()
            abs_pos = root.translate_coords(window, 0, 0)
            ret = {"left": abs_pos.x, "top": abs_pos.y, "width": geom.width, "height": geom.height}
        except:  # noqa # Window not found falling back to initial window area
            ret = {}
            pos = comp[-3].split(',')
            ret["left"] = int(pos[0])
            ret["top"] = int(pos[1])
            pos = comp[-2].split(',')
            ret["width"] = int(pos[0])
            ret["height"] = int(pos[1])
        # Ensure bounding box is stricly inside monitor area
        if ret["left"] < 0:
            ret["width"] += ret["left"]
            ret["left"] = 0
        if ret["top"] < 0:
            ret["height"] += ret["top"]
            ret["top"] = 0
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!")
            return dict(_mss.monitors[1])  # Return primary monitor area
        box = _mss.monitors[0]  # Combined monitor area
        if ret["left"] + ret["width"] >= box["width"]:
            ret["width"] = box["width"] - ret["left"] - 1
        if ret["top"] + ret["height"] >= box["height"]:
            ret["height"] = box["height"] - ret["top"] - 1
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!")
            return dict(_mss.monitors[1])  # Return primary monitor area
        return ret
    else:
        return {"left": 0, "right": 0, "width": 0, "height": 0}  # WARNING: mac is unimplemented!


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


def capture_process(frame_size, monitor_index, window_name, frame_shm, frame_lock, frame_event, stop_event, backend="mss", crop_top=0, crop_left=0, crop_right=0, crop_bottom=0):
    frame_buffer = np.ndarray(frame_size, dtype=np.uint8, buffer=frame_shm.buf)
    frame_count = 0

    if backend == "mss":
        capture = WindowsCaptureMSS(monitor_index=monitor_index, window_name=window_name)
    elif backend == "windows_capture":
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            frame_event.set()
            raise

        if window_name:
            # ignore
            monitor_index = None
        else:
            # 1 origin
            monitor_index = monitor_index + 1

        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=monitor_index,
            window_name=window_name,
        )

    @capture.event
    def on_frame_arrived(frame, capture_control):
        nonlocal frame_shm, frame_event, frame_lock, stop_event, frame_buffer, window_name, frame_count, crop_top, crop_left, crop_right, crop_bottom  # noqa
        if not frame_event.is_set():
            with frame_lock:
                source_frame = frame.frame_buffer
                # Apply cropping for window capture
                if crop_top > 0 or crop_left > 0 or crop_right > 0 or crop_bottom > 0:
                    h, w = source_frame.shape[:2]
                    top = crop_top
                    bottom = h - crop_bottom if crop_bottom > 0 else h
                    left = crop_left
                    right = w - crop_right if crop_right > 0 else w
                    source_frame = source_frame[top:bottom, left:right, :]

                if frame_buffer.shape != source_frame.shape:
                    if window_name is not None:
                        # NOTE: The size may differ due to resizing, Window effects, etc.
                        # I wanted to use replication padding, but since the edges of the Windows screen are black, it's meaningless
                        min_h = min(frame_buffer.shape[0], source_frame.shape[0])
                        min_w = min(frame_buffer.shape[1], source_frame.shape[1])
                        if frame_count % 30 == 0:
                            frame_buffer[:] = 0.0
                        frame_buffer[0:min_h, 0:min_w, :] = source_frame[0:min_h, 0:min_w, :]
                        frame_count += 1
                        if frame_count > 0xffff:
                            frame_count = 0
                    else:
                        raise RuntimeError(f"Screen size missmatch. frame_buffer={frame_buffer.shape}, frame={source_frame.shape}")
                else:
                    frame_buffer[:] = source_frame
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
        time.sleep(0.1)


def to_tensor(bgra, device):
    x = torch.from_numpy(bgra)
    x = x[:, :, :3].permute(2, 0, 1).contiguous().to(device)
    x = x / 255.0
    return x


class ScreenshotProcess(threading.Thread):
    def __init__(self, fps, frame_width, frame_height, monitor_index, window_name, device, backend="mss",
                 crop_top=0, crop_left=0, crop_right=0, crop_bottom=0, fullscreen_framebuf=False, ar_preserve=True):
        super().__init__(daemon=True)
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.device = device
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.fullscreen_framebuf = fullscreen_framebuf
        self.ar_preserve = ar_preserve
        self.frame = None
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)

        if device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=device)
        else:
            self.cuda_stream = None

    def get_capture_size(self):
        if self.window_name:
            rect = get_window_rect_by_title(self.window_name)
            if rect is None:
                raise RuntimeError(f"{self.window_name} not found")
        else:
            rect = dict(_mss.monitors[self.monitor_index + 1])  # deep copy
        rect["top"] += self.crop_top
        rect["left"] += self.crop_left
        rect["height"] -= self.crop_top + self.crop_bottom
        rect["width"] -= self.crop_left + self.crop_right
        if rect["width"] < 64 or rect["height"] < 64:
            raise RuntimeError(f"capture region ={str(rect)} is too small (below 64 x 64)")
        return rect

    def start_process(self):
        capture_size = self.get_capture_size()
        self.screen_width = capture_size["width"]
        self.screen_height = capture_size["height"]
        if self.fullscreen_framebuf:
            self.screen_width = _mss.monitors[self.monitor_index + 1]["width"] - self.crop_left - self.crop_right
            self.screen_height = _mss.monitors[self.monitor_index + 1]["height"] - self.crop_top - self.crop_bottom
        if self.screen_width < 64 or self.screen_height < 64:
            raise RuntimeError(f"capture size ={self.frame_width},{self.frame_height} is too small (below 64 x 64)")
        template = np.zeros((self.screen_height, self.screen_width, 4), dtype=np.uint8)
        self.process_frame_buffer = shared_memory.SharedMemory(create=True, size=template.nbytes)
        self.process_stop_event = mp.Event()
        self.process_frame_event = mp.Event()
        self.process_frame_lock = mp.Lock()
        self.process = mp.Process(
            target=capture_process,
            args=(tuple(template.shape),
                  self.monitor_index,
                  self.window_name,
                  self.process_frame_buffer,
                  self.process_frame_lock,
                  self.process_frame_event,
                  self.process_stop_event,
                  self.backend,
                  self.crop_top,
                  self.crop_left,
                  self.crop_right,
                  self.crop_bottom))
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

                capture_size = self.get_capture_size()
                min_h = min(capture_size["height"], self.screen_height)
                min_w = min(capture_size["width"], self.screen_width)
                if self.ar_preserve:
                    if min_h * self.screen_width // self.screen_height > min_w:
                        min_w = min_h * self.screen_width // self.screen_height
                    elif min_w * self.screen_height // self.screen_width > min_h:
                        min_h = min_w * self.screen_height // self.screen_width
                height_fit = min_h < self.screen_height
                width_fit = min_w < self.screen_width
                perfect_fit = width_fit and height_fit
                partial_fit = width_fit or height_fit
                crop_needed = ((not self.ar_preserve) and partial_fit) or perfect_fit

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        if self.backend == "mss":
                            # cursor for MSS
                            draw_cursor(frame, wx.GetMousePosition(), capture_size)
                        if crop_needed:
                            frame = TF.crop(frame, 0, 0, min_h, min_w)
                        if frame.shape[1:] != (self.frame_height, self.frame_width):
                            frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                              interpolation=InterpolationMode.BILINEAR,
                                              antialias=True)
                        self.cuda_stream.synchronize()
                else:
                    frame = frame_buffer.to(self.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    if self.backend == "mss":
                        draw_cursor(frame, wx.GetMousePosition(), capture_size)
                    if crop_needed:
                        frame = TF.crop(frame, 0, 0, min_h, min_w)
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
            self.stop_event.set()
            self.process.join(timeout=4)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.process = None
            self.process_frame_buffer.close()
            self.process_frame_buffer.unlink()
            self.process_frame_buffer = None

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
            self.join(timeout=4)
