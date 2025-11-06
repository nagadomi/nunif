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


_x11_connection_pool = {}
_mss_pool = {}


def get_x11():
    key = os.getpid()
    display, root = _x11_connection_pool.get(key, (None, None))
    if display is None:
        from Xlib import display as Xdisplay
        display = Xdisplay.Display()
        root = display.screen().root
        _x11_connection_pool[key] = (display, root)

    return display, root


def get_x11root():
    display, root = get_x11()
    return root


def get_mss():
    key = os.getpid()
    sct = _mss_pool.get(key)
    if sct is None:
        import mss
        sct = mss.mss(with_cursor=True)
        _mss_pool[key] = sct

    return sct


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
        import mss
        control = CaptureControlMSS()
        while True:
            with mss.mss(with_cursor=True) as sct:
                tick = time.perf_counter()
                if self.window_name is not None:
                    position = get_window_rect_by_title(self.window_name, sct=sct)
                else:
                    position = sct.monitors[self.monitor_index + 1]
                shot = sct.grab(position)
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
        monitors = get_mss().monitors[1:]
        size_list = []
        for monitor in monitors:
            size_list.append((monitor["width"], monitor["height"]))
        return size_list


def get_screen_size(monitor_index):
    size_list = get_monitor_size_list()
    return size_list[monitor_index]


DENY_WINDOW_NAMES = {
    "Microsoft Text Input Application",
    "Program Manager"
}


def enum_window_names_x11():
    from Xlib import X, Xatom
    d, root = get_x11()

    NET_CLIENT_LIST = d.intern_atom("_NET_CLIENT_LIST")
    NET_WM_NAME = d.intern_atom("_NET_WM_NAME")
    NET_FRAME_EXTENTS = d.intern_atom("_NET_FRAME_EXTENTS")
    NET_WM_DESKTOP = d.intern_atom("_NET_WM_DESKTOP")
    NET_CURRENT_DESKTOP = d.intern_atom("_NET_CURRENT_DESKTOP")
    UTF8_STRING = d.intern_atom("UTF8_STRING")

    current_desktop_prop = root.get_full_property(NET_CURRENT_DESKTOP, X.AnyPropertyType)
    current_desktop = current_desktop_prop.value[0] if current_desktop_prop else None

    client_list_prop = root.get_full_property(NET_CLIENT_LIST, X.AnyPropertyType)
    if not client_list_prop:
        return []

    windows = []
    for win_id in client_list_prop.value:
        try:
            w = d.create_resource_object("window", win_id)
            desk_prop = w.get_full_property(NET_WM_DESKTOP, X.AnyPropertyType)
            if desk_prop:
                desktop = desk_prop.value[0]
                if current_desktop is not None and desktop != current_desktop:
                    continue

            name = None
            for atom in (NET_WM_NAME, Xatom.WM_NAME):
                prop = w.get_full_property(atom, UTF8_STRING)
                if prop:
                    name = prop.value
                    if isinstance(name, bytes):
                        name = name.decode(errors="ignore")
                    break
            if not name:
                continue

            geom = w.get_geometry()
            abs_pos = root.translate_coords(w, 0, 0)
            x, y = abs_pos.x, abs_pos.y
            width, height = geom.width, geom.height
            if width < 128 or height < 128:
                continue

            window_name = (str(name) + "|" + str(abs_pos.x) + "," + str(abs_pos.y) + "|" +
                           str(geom.width) + "," + str(geom.height) + "|" + str(w.id))
            windows.append(window_name)

        except Xlib.error.XError:
            continue

    return sorted(windows)


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
        return enum_window_names_x11()
    else:
        return []  # WARNING: mac is unimplemented!


def find_window_x11(address, x_display):
    try:
        window = x_display.create_resource_object("window", int(address))
    except Xlib.error.XError:
        window = None

    return window


def get_window_rect_by_title(title, sct=None):
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
        x_display, x_root = get_x11()
        comp = title.rsplit("|", 4)
        if len(comp) < 4:
            return None

        window = find_window_x11(comp[-1], x_display)
        if window is not None:
            geom = window.get_geometry()
            abs_pos = x_root.translate_coords(window, 0, 0)
            ret = {"left": abs_pos.x, "top": abs_pos.y, "width": geom.width, "height": geom.height}
        else:
            # Window not found falling back to initial window area
            ret = {}
            pos = comp[-3].split(',')
            ret["left"] = int(pos[0])
            ret["top"] = int(pos[1])
            size = comp[-2].split(',')
            ret["width"] = int(size[0])
            ret["height"] = int(size[1])

        # Ensure bounding box is stricly inside monitor area
        sct = sct or get_mss()

        if ret["left"] < 0:
            ret["width"] += ret["left"]
            ret["left"] = 0
        if ret["top"] < 0:
            ret["height"] += ret["top"]
            ret["top"] = 0
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!", file=sys.stderr)
            # Return primary monitor area
            return dict(sct.monitors[1])

        box = sct.monitors[0]  # Combined monitor area
        if ret["left"] + ret["width"] >= box["width"]:
            ret["width"] = box["width"] - ret["left"] - 1
        if ret["top"] + ret["height"] >= box["height"]:
            ret["height"] = box["height"] - ret["top"] - 1
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!", file=sys.stderr)
            # Return primary monitor area
            return dict(sct.monitors[1])

        return ret
    else:
        # TODO: Not implemented
        return {"left": 0, "right": 0, "width": 0, "height": 0}


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
                if window_name and (crop_top > 0 or crop_left > 0 or crop_right > 0 or crop_bottom > 0):
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
                 crop_top=0, crop_left=0, crop_right=0, crop_bottom=0):
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
        if self.window_name:
            rect = get_window_rect_by_title(self.window_name)
            if rect is None:
                raise RuntimeError(f"{self.window_name} not found")
            screen_size = (rect["width"], rect["height"])
        else:
            screen_size = get_screen_size(self.monitor_index)
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

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        if self.backend == "mss":
                            # cursor for MSS
                            # On Linux, it is implmented in mss
                            # TODO: windows
                            # draw_cursor(frame, wx.GetMousePosition())
                            pass
                        if frame.shape[1:] != (self.frame_height, self.frame_width):
                            frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                              interpolation=InterpolationMode.BILINEAR,
                                              antialias=True)
                        self.cuda_stream.synchronize()
                else:
                    frame = frame_buffer.to(self.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    if self.backend == "mss":
                        # on Linux, it is implmented in mss
                        # TODO: windows
                        # draw_cursor(frame, wx.GetMousePosition())
                        pass
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
