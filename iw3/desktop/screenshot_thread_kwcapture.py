"""Wayland (KDE Plasma) screenshot backend, through kwcapture.

Why this exists: on a Wayland session every other backend here is wrong. ``mss`` and
``PIL.ImageGrab`` capture the XWayland root window -- black for native Wayland windows, one
screen only, no per-window capture (and ``is_mss_supported()`` refuses Wayland outright),
while ``windows_capture``/``wc_cuda`` are Windows only. kwcapture asks KWin itself for the
image (``org.kde.KWin.ScreenShot2``), so it gets the real composited frame at device
resolution, per monitor and per window, at roughly 40 fps at 1440p -- no portal, no root
access, no XWayland.

See https://pypi.org/project/kwcapture/ (also
https://github.com/tjandrasg/kwcapture). kwcapture is an **optional** dependency: it is not in
``requirements-gui.txt``, because it only works on one compositor and the user may not want it.
Install it with ``pip install kwcapture`` when needed; the wheel ships the small native helper
that KWin authorises through a KDE desktop entry, which kwcapture installs by itself on first
use.

Everything in this module degrades gracefully: importing it never requires kwcapture,
``is_kwcapture_supported()`` is cheap enough to call from the GUI at startup, and
``warn_if_kwcapture_missing()`` tells the user about the optional package only on a session
where it would actually have helped (KDE Plasma on Wayland).
"""
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)


_kwcapture = None
_kwcapture_checked = False
_missing_warned = False


def is_wayland():
    return (sys.platform == "linux" and
            os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland")


def is_kde_desktop():
    """True when the session advertises KDE Plasma (the only desktop kwcapture can talk to).

    Used for the *warning* only, never to offer the backend: whether kwcapture works is
    decided by the presence of the package plus a Wayland session, and a heuristic must not
    hide a feature from somebody who did install it.
    """
    if sys.platform != "linux":
        return False
    session = (os.environ.get("XDG_CURRENT_DESKTOP", "") + ":" +
               os.environ.get("DESKTOP_SESSION", "")).lower()
    return "kde" in session or "plasma" in session


def get_kwcapture():
    """The kwcapture module, or None when it is not installed / cannot be loaded."""
    global _kwcapture, _kwcapture_checked
    if not _kwcapture_checked:
        _kwcapture_checked = True
        try:
            import kwcapture
            _kwcapture = kwcapture
        except ImportError:
            _kwcapture = None
    return _kwcapture


def kwcapture_unsupported_reason():
    """Why kwcapture cannot be used here, or None when it can (cheap: no daemon started).

    A KWin that refuses the helper still fails later, with kwcapture's own message; this only
    decides whether the backend is offered at all.
    """
    if get_kwcapture() is None:
        return "kwcapture is not installed (pip install kwcapture)"
    if not is_wayland():
        return ("kwcapture needs a Wayland session, but XDG_SESSION_TYPE="
                f"{os.environ.get('XDG_SESSION_TYPE', 'unset')!r}")
    return None


def is_kwcapture_supported():
    """True when this session can plausibly use kwcapture (cheap: no daemon is started)."""
    return kwcapture_unsupported_reason() is None


def is_kwcapture_install_recommended():
    """True when kwcapture would work here and is not installed.

    This is the only situation worth warning about: an optional dependency the user cannot
    use at all (not a KDE session, X11) should stay quiet.
    """
    return is_kde_desktop() and is_wayland() and get_kwcapture() is None


# Single source of the text, so the console message and the GUI translation cannot drift
# apart. Also used as the locale key in iw3/locales/*.yml.
KWCAPTURE_MISSING_MESSAGE = (
    "kwcapture is not installed, so this Wayland (KDE Plasma) desktop cannot be captured. "
    "Install it with `pip install kwcapture`. The other screenshot methods use XWayland, "
    "which is usually black on Wayland.")


def warn_if_kwcapture_missing():
    """Warn once per process that the optional kwcapture package is missing, if it applies."""
    global _missing_warned
    if not _missing_warned and is_kwcapture_install_recommended():
        _missing_warned = True
        print(f"Warning: {KWCAPTURE_MISSING_MESSAGE}", file=sys.stderr)
        return True
    return False


def get_monitor_list():
    """kwcapture Monitor objects (top-left first), or [] when unavailable."""
    K = get_kwcapture()
    if K is None or not is_wayland():
        return []
    try:
        return K.list_monitors()
    except Exception as e: # noqa
        print(f"kwcapture: cannot list monitors: {e}", file=sys.stderr)
        return []


def get_monitor_size_list():
    """[(width, height)] in device pixels -- same order as mss' ``monitors[1:]``."""
    return [(m.width, m.height) for m in get_monitor_list()]


def get_monitor_scale_list():
    """[(display scale, scene factor)] per monitor, both measured by kwcapture.

    A Wayland output's logical size differs from its pixel size by the display scale, and
    region captures are rendered at the scene factor, so "pixels" only means something
    once you know which of the two you are holding.
    """
    K = get_kwcapture()
    if K is None or not is_wayland():
        return []
    try:
        return [(m.effective_scale, m.area_scale) for m in K.list_monitors(measure_scale=True)]
    except Exception as e: # noqa
        print(f"kwcapture: cannot measure scales: {e}", file=sys.stderr)
        return []


def enum_window_names():
    """Capturable window captions, sorted and de-duplicated (like the Windows backend)."""
    K = get_kwcapture()
    if K is None or not is_wayland():
        return []
    try:
        windows = K.list_windows()
    except Exception as e: # noqa
        print(f"kwcapture: cannot list windows: {e}", file=sys.stderr)
        return []
    names = set()
    for w in windows:
        if w.width < 128 or w.height < 128:
            continue  # tooltips, popups, input-method windows
        if w.name:
            names.add(w.name)
    return sorted(names)


def resolve_window(name, K=None):
    """The kwcapture Window for a caption; the largest match wins when several agree."""
    K = K or get_kwcapture()
    if K is None:
        raise RuntimeError("kwcapture is not installed (pip install kwcapture)")
    try:
        return K.find_window(name)
    except K.AmbiguousWindow:
        # find_window() refuses to guess; the GUI list is de-duplicated, so a tie means two
        # windows really do share a caption. Take the biggest visible one.
        candidates = [w for w in K.list_windows() if w.name == name]
        if not candidates:
            raise
        return max(candidates, key=lambda w: w.width * w.height)


def get_window_rect_by_title(title):
    """{"left", "top", "width", "height"} of a window, in **device** pixels.

    The size is measured from a real frame rather than from KWin's window geometry, because
    that geometry is in *logical* pixels: on a display scaled to 125 % a window listed as
    472x466 is captured as 590x583 (and at 75 % as 354x329). iw3 sizes its stream from
    these numbers, so they have to be the pixels the capture thread will produce.
    """
    K = get_kwcapture()
    if K is None or not is_wayland():
        return None
    try:
        window = resolve_window(title, K)
    except Exception as e: # noqa
        print(f"kwcapture: window {title} not found: {e}", file=sys.stderr)
        return None
    try:
        with K.Capture(window=window.id, shm=K.unique_shm_path("iw3rect")) as capture:
            capture.grab(copy=True, timeout=10)
            width, height = capture.geometry()
    except Exception as e: # noqa
        print(f"kwcapture: cannot measure window {title}: {e}", file=sys.stderr)
        return None
    return {"left": window.x, "top": window.y, "width": width, "height": height}


def to_tensor(bgra, device, frame_buffer, non_blocking=False):
    """BGRA (H, W, 4) uint8 -> RGB CHW float [0, 1] on `device`, through `frame_buffer`.

    `frame_buffer` is the (pinned, when CUDA is around) host buffer the frames are staged
    in; copying into it is synchronous, so the async H2D transfer afterwards is safe.
    """
    x = frame_buffer.copy_(torch.from_numpy(bgra))
    x = x.to(device, non_blocking=non_blocking)
    x = x[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous()  # BGRA -> RGB, CHW
    return x / 255.0


class ScreenshotThreadKWCapture(threading.Thread):
    """Frames from KWin through kwcapture, for a whole monitor or one window.

    Resizes and mode changes need no attention: kwcapture follows the target and simply
    returns frames at the new size, and this thread keeps its buffers sized to match.
    The pointer is included by the compositor itself when ``draw_cursor_enabled`` is set,
    so no marker is painted over the image.
    """

    def __init__(self, fps, frame_width, frame_height, monitor_index, window_name, device,
                 crop_top=0, crop_left=0, crop_right=0, crop_bottom=0,
                 draw_cursor_enabled=True, **_ignore_unsupported_kwargs):
        super().__init__(daemon=True)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.device = device
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.draw_cursor_enabled = draw_cursor_enabled
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.frame = None
        self.error = None
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        self.capture = None
        self.monitor_name = None
        if device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=device)
        else:
            self.cuda_stream = None
        # Host->device copies overlap only from pinned host memory, so pin when the frames are
        # going to CUDA. Never otherwise (see run()).
        self.pinned_buffer = self.cuda_stream is not None

    def create_capture(self):
        reason = kwcapture_unsupported_reason()
        if reason is not None:
            # Same wording as the CLI check, so a thread built by the GUI fails with the
            # message that says what to do about it.
            raise RuntimeError(reason)
        K = get_kwcapture()
        kwargs = {"cursor": self.draw_cursor_enabled, "shm": K.unique_shm_path("iw3")}
        if self.window_name:
            window = resolve_window(self.window_name, K)
            kwargs["window"] = window.id
            self.monitor_name = window.name
        else:
            monitors = K.list_monitors()
            if not monitors:
                raise RuntimeError("kwcapture found no output on this session")
            if self.monitor_index >= len(monitors):
                raise RuntimeError(f"monitor_index={self.monitor_index} not found "
                                   f"(this session has {len(monitors)})")
            kwargs["monitor"] = monitors[self.monitor_index]
            self.monitor_name = monitors[self.monitor_index].name
        return K.Capture(**kwargs)

    def crop(self, bgra):
        if not (self.crop_top or self.crop_left or self.crop_right or self.crop_bottom):
            return bgra
        h, w = bgra.shape[:2]
        top = min(self.crop_top, max(h - 1, 0))
        left = min(self.crop_left, max(w - 1, 0))
        bottom = h - self.crop_bottom if self.crop_bottom > 0 else h
        right = w - self.crop_right if self.crop_right > 0 else w
        return bgra[top:bottom, left:right, :]

    def run(self):
        frame_buffer = None
        try:
            capture = self.create_capture()
            self.capture = capture
            while True:
                tick = time.perf_counter()
                # copy=True on purpose: the zero-copy frame is a view of a ring slot, and a
                # slow consumer (the depth model) must not be able to read a slot that the
                # compositor is rewriting.
                bgra = self.crop(capture.grab(copy=True, timeout=10))
                shape = (bgra.shape[0], bgra.shape[1], 4)
                if frame_buffer is None or frame_buffer.shape != shape:
                    frame_buffer = torch.from_numpy(np.ascontiguousarray(bgra))
                    if self.pinned_buffer:
                        # Only worth it when the frames are headed for a CUDA device: pinning
                        # initializes the CUDA context, which costs a few hundred MB of VRAM and
                        # fails outright on a card that is already full -- for a run that asked
                        # for a CPU device and never needed CUDA at all.
                        try:
                            frame_buffer = frame_buffer.pin_memory()
                        except Exception as e: # noqa
                            self.pinned_buffer = False
                            print(f"kwcapture: cannot pin the frame buffer ({e}),"
                                  " using pageable memory instead", file=sys.stderr)
                else:
                    frame_buffer.copy_(torch.from_numpy(bgra))

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = to_tensor(bgra, self.device, frame_buffer, non_blocking=True)
                        frame = self.resize(frame)
                    frame.record_stream(self.cuda_stream)
                else:
                    frame = self.resize(to_tensor(bgra, self.device, frame_buffer))

                with self.frame_lock:
                    self.frame = frame
                    self.frame_set_event.set()

                with self.fps_lock:
                    self.fps_counter.append(time.perf_counter() - tick)

                if self.stop_event.is_set():
                    break
                if self.fps and self.fps > 0:
                    wait = (1.0 / self.fps) - (time.perf_counter() - tick)
                    if wait > 0:
                        self.stop_event.wait(wait)
        except Exception as e: # noqa
            self.error = e
        finally:
            if self.capture is not None:
                try:
                    self.capture.close()
                except Exception: # noqa
                    pass
                self.capture = None
            self.frame_set_event.set()  # let get_frame() report the failure

    def resize(self, frame):
        if frame.shape[1:] != (self.frame_height, self.frame_width):
            frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                              interpolation=InterpolationMode.BILINEAR, antialias=True)
        return frame

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if not self.is_alive():
                break
        if self.error is not None:
            raise RuntimeError(f"kwcapture screenshot thread failed: {self.error}") from self.error
        if not self.frame_set_event.is_set():
            raise RuntimeError("thread is already dead")
        with self.frame_lock:
            if self.cuda_stream is not None:
                torch.cuda.current_stream().wait_stream(self.cuda_stream)
            frame = self.frame
            self.frame = None
            self.frame_set_event.clear()
        assert frame is not None
        return frame

    def get_fps(self):
        with self.fps_lock:
            if self.fps_counter:
                return 1 / (sum(self.fps_counter) / len(self.fps_counter))
            return 0

    def stop(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout=4)
