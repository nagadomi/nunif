# NOTE: libobs runs in a separate worker process (spawn):
# - libobs cannot be re-initialized in the same process after obs_shutdown(),
#   and obs_shutdown() with the PipeWire/GL context can crash or hang on teardown.
# - Avoid loading libobs (GPL) into the main process.

import ctypes
import ctypes.util
import multiprocessing as mp
import os
import sys
import sysconfig
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as MpEvent
from multiprocessing.synchronize import Lock as MpLock
from typing import Any

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_obs_support_checked = False
_obs_unsupported_reason: str | None = None


def is_wayland() -> bool:
    if sys.platform != "linux":
        return False
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def _get_library_search_dirs() -> list[str]:
    dirs: list[str] = []
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        dirs.append(f"/usr/lib/{multiarch}")
        dirs.append(f"/usr/local/lib/{multiarch}")
    dirs.extend(
        [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/usr/local/lib64",
            "/usr/local/lib",
        ]
    )
    seen: set[str] = set()
    unique_dirs: list[str] = []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def _find_shared_library(name: str, sonames: tuple[str, ...]) -> str | None:
    found = ctypes.util.find_library(name)
    if found:
        return found
    for lib_dir in _get_library_search_dirs():
        for soname in sonames:
            candidate = os.path.join(lib_dir, soname)
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_pipewire_plugin() -> str | None:
    env_path = os.environ.get("OBS_PIPEWIRE_PLUGIN_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    for lib_dir in _get_library_search_dirs():
        candidate = os.path.join(lib_dir, "obs-plugins", "linux-pipewire.so")
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_obs_paths() -> tuple[dict[str, str] | None, str | None]:
    if sys.platform != "linux":
        return None, "libobs PipeWire capture is only supported on Linux"
    if not is_wayland():
        return None, "libobs PipeWire capture requires a Wayland session"

    libobs = _find_shared_library("obs", ("libobs.so.30", "libobs.so.0", "libobs.so"))
    if not libobs:
        return None, "libobs shared library not found (install obs-studio / libobs)"

    libobs_opengl = _find_shared_library(
        "obs-opengl", ("libobs-opengl.so.30", "libobs-opengl.so.1", "libobs-opengl.so")
    )
    if not libobs_opengl:
        return None, "libobs-opengl shared library not found (install libobs)"

    libglib = _find_shared_library("glib-2.0", ("libglib-2.0.so.0", "libglib-2.0.so"))
    if not libglib:
        return None, "libglib-2.0 shared library not found"

    libwayland = _find_shared_library("wayland-client", ("libwayland-client.so.0", "libwayland-client.so"))
    if not libwayland:
        return None, "libwayland-client shared library not found"

    pipewire_plugin = _find_pipewire_plugin()
    if not pipewire_plugin:
        return None, "OBS linux-pipewire.so plugin not found (install obs-plugins)"

    return {
        "libobs": libobs,
        "libobs_opengl": libobs_opengl,
        "libglib": libglib,
        "libwayland": libwayland,
        "pipewire_plugin": pipewire_plugin,
    }, None


def obs_unsupported_reason() -> str | None:
    """Return None if OBS PipeWire backend is available, or a reason string otherwise.

    Does not dlopen/load GPL libobs into the main process; only checks environment and files.
    """
    global _obs_support_checked, _obs_unsupported_reason
    if not _obs_support_checked:
        _obs_support_checked = True
        _, _obs_unsupported_reason = _resolve_obs_paths()
    return _obs_unsupported_reason


def is_obs_supported() -> bool:
    return obs_unsupported_reason() is None


def _obs_capture_worker(
    conn: Connection,
    lib_paths: dict[str, str],
    fps: int,
    draw_cursor_enabled: bool,
    parent_pid: int,
    frame_lock: MpLock,
    frame_event: MpEvent,
    stop_event: MpEvent,
) -> None:
    """Isolated worker process that loads libobs (GPL) and captures frames via PipeWire."""
    shm: shared_memory.SharedMemory | None = None
    try:
        libobs = ctypes.CDLL(lib_paths["libobs"])
        libglib = ctypes.CDLL(lib_paths["libglib"])
        libwayland = ctypes.CDLL(lib_paths["libwayland"])
    except OSError as e:
        conn.send({"status": "error", "message": f"Failed to load shared libraries: {e}"})
        os._exit(1)

    OBS_NIX_PLATFORM_WAYLAND = 2
    VIDEO_FORMAT_BGRA = 7
    VIDEO_CS_DEFAULT = 0
    VIDEO_RANGE_FULL = 2
    OBS_SCALE_BILINEAR = 3
    MAX_AV_PLANES = 8

    class obs_video_info(ctypes.Structure):
        _fields_ = [
            ("graphics_module", ctypes.c_char_p),
            ("fps_num", ctypes.c_uint32),
            ("fps_den", ctypes.c_uint32),
            ("base_width", ctypes.c_uint32),
            ("base_height", ctypes.c_uint32),
            ("output_width", ctypes.c_uint32),
            ("output_height", ctypes.c_uint32),
            ("output_format", ctypes.c_int),
            ("adapter", ctypes.c_uint32),
            ("gpu_conversion", ctypes.c_bool),
            ("colorspace", ctypes.c_int),
            ("range", ctypes.c_int),
            ("scale_type", ctypes.c_int),
        ]

    class video_scale_info(ctypes.Structure):
        _fields_ = [
            ("format", ctypes.c_int),
            ("width", ctypes.c_uint32),
            ("height", ctypes.c_uint32),
            ("range", ctypes.c_int),
            ("colorspace", ctypes.c_int),
        ]

    class video_data(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.POINTER(ctypes.c_uint8) * MAX_AV_PLANES),
            ("linesize", ctypes.c_uint32 * MAX_AV_PLANES),
            ("timestamp", ctypes.c_uint64),
        ]

    RAW_VIDEO_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(video_data))
    LOG_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)

    # GLib functions
    libglib.g_main_context_default.argtypes = []
    libglib.g_main_context_default.restype = ctypes.c_void_p
    libglib.g_main_context_iteration.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libglib.g_main_context_iteration.restype = ctypes.c_int

    # Wayland functions
    libwayland.wl_display_connect.argtypes = [ctypes.c_char_p]
    libwayland.wl_display_connect.restype = ctypes.c_void_p
    libwayland.wl_display_disconnect.argtypes = [ctypes.c_void_p]
    libwayland.wl_display_disconnect.restype = None

    # OBS functions
    libobs.base_set_log_handler.argtypes = [LOG_HANDLER, ctypes.c_void_p]
    libobs.base_set_log_handler.restype = None

    libobs.obs_set_nix_platform.argtypes = [ctypes.c_int]
    libobs.obs_set_nix_platform.restype = None
    libobs.obs_set_nix_platform_display.argtypes = [ctypes.c_void_p]
    libobs.obs_set_nix_platform_display.restype = None

    libobs.obs_startup.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    libobs.obs_startup.restype = ctypes.c_bool
    libobs.obs_reset_video.argtypes = [ctypes.POINTER(obs_video_info)]
    libobs.obs_reset_video.restype = ctypes.c_int

    libobs.obs_open_module.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p]
    libobs.obs_open_module.restype = ctypes.c_int
    libobs.obs_init_module.argtypes = [ctypes.c_void_p]
    libobs.obs_init_module.restype = ctypes.c_bool
    libobs.obs_post_load_modules.argtypes = []
    libobs.obs_post_load_modules.restype = None

    libobs.obs_data_create.argtypes = []
    libobs.obs_data_create.restype = ctypes.c_void_p
    libobs.obs_data_set_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
    libobs.obs_data_set_bool.restype = None
    libobs.obs_data_release.argtypes = [ctypes.c_void_p]
    libobs.obs_data_release.restype = None

    libobs.obs_source_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
    libobs.obs_source_create.restype = ctypes.c_void_p
    libobs.obs_source_get_width.argtypes = [ctypes.c_void_p]
    libobs.obs_source_get_width.restype = ctypes.c_uint32
    libobs.obs_source_get_height.argtypes = [ctypes.c_void_p]
    libobs.obs_source_get_height.restype = ctypes.c_uint32

    libobs.obs_set_output_source.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    libobs.obs_set_output_source.restype = None

    libobs.obs_add_raw_video_callback.argtypes = [
        ctypes.POINTER(video_scale_info),
        RAW_VIDEO_CALLBACK,
        ctypes.c_void_p,
    ]
    libobs.obs_add_raw_video_callback.restype = None
    libobs.obs_remove_raw_video_callback.argtypes = [RAW_VIDEO_CALLBACK, ctypes.c_void_p]
    libobs.obs_remove_raw_video_callback.restype = None

    portal_error: str | None = None

    @LOG_HANDLER
    def on_obs_log(level: int, msg: bytes, args_ptr: int, param: int) -> None:
        nonlocal portal_error
        if not msg:
            return
        if (
            b"Request cancelled" in msg
            or b"denied or cancelled by user" in msg
            or b"[pipewire] Failed to " in msg
            or b"[pipewire] Error " in msg
            or b"No capture sources available" in msg
        ):
            portal_error = msg.decode("utf-8", errors="replace")

    libobs.base_set_log_handler(on_obs_log, None)

    wl_display = libwayland.wl_display_connect(None)
    if not wl_display:
        conn.send({"status": "error", "message": "Failed to connect to Wayland display"})
        os._exit(1)

    libobs.obs_set_nix_platform(OBS_NIX_PLATFORM_WAYLAND)
    libobs.obs_set_nix_platform_display(wl_display)

    if not libobs.obs_startup(b"en-US", None, None):
        conn.send({"status": "error", "message": "obs_startup failed"})
        os._exit(1)

    capture_fps = max(int(fps), 1) if fps and fps > 0 else 60
    probe_fps = max(capture_fps, 30)
    opengl_so_bytes = lib_paths["libobs_opengl"].encode("utf-8")

    # Initial dummy video reset to initialize the OpenGL graphics context required by linux-pipewire.
    ovi = obs_video_info(
        graphics_module=opengl_so_bytes,
        fps_num=probe_fps,
        fps_den=1,
        base_width=1920,
        base_height=1080,
        output_width=1920,
        output_height=1080,
        output_format=VIDEO_FORMAT_BGRA,
        adapter=0,
        gpu_conversion=True,
        colorspace=VIDEO_CS_DEFAULT,
        range=VIDEO_RANGE_FULL,
        scale_type=OBS_SCALE_BILINEAR,
    )
    res = libobs.obs_reset_video(ctypes.byref(ovi))
    if res != 0:
        conn.send({"status": "error", "message": f"Initial obs_reset_video failed (code={res})"})
        os._exit(1)

    mod = ctypes.c_void_p()
    res = libobs.obs_open_module(ctypes.byref(mod), lib_paths["pipewire_plugin"].encode("utf-8"), None)
    if res != 0 or not mod.value:
        conn.send({"status": "error", "message": f"obs_open_module(linux-pipewire) failed (code={res})"})
        os._exit(1)

    if not libobs.obs_init_module(mod):
        conn.send({"status": "error", "message": "obs_init_module(linux-pipewire) failed"})
        os._exit(1)
    libobs.obs_post_load_modules()

    settings = libobs.obs_data_create()
    libobs.obs_data_set_bool(settings, b"ShowCursor", bool(draw_cursor_enabled))

    source = libobs.obs_source_create(b"pipewire-screen-capture-source", b"screen_cap", settings, None)
    libobs.obs_data_release(settings)
    if not source:
        conn.send({"status": "error", "message": "obs_source_create(pipewire-screen-capture-source) failed"})
        os._exit(1)

    libobs.obs_set_output_source(0, source)

    # Wait until the first non-empty frame is actually rendered so that PipeWire
    # stream negotiation and per-buffer SPA_META_VideoCrop metadata are applied
    # before querying obs_source_get_width/height.
    probe_w = 1920
    probe_h = 1080
    probe_done = False

    @RAW_VIDEO_CALLBACK
    def on_probe_frame(param: int, frame_ptr: Any) -> None:
        nonlocal probe_done
        if probe_done or stop_event.is_set() or not frame_ptr:
            return
        frame = frame_ptr.contents
        linesize = frame.linesize[0]
        data_ptr = frame.data[0]
        if not data_ptr or linesize < probe_w * 4:
            return

        arr = np.ctypeslib.as_array(data_ptr, shape=(probe_h, linesize))
        bgra = arr[:, : probe_w * 4].reshape(probe_h, probe_w, 4)
        if not np.any(bgra > 0):
            return
        probe_done = True

    probe_scale_info = video_scale_info(
        format=VIDEO_FORMAT_BGRA,
        width=probe_w,
        height=probe_h,
        range=VIDEO_RANGE_FULL,
        colorspace=VIDEO_CS_DEFAULT,
    )
    libobs.obs_add_raw_video_callback(ctypes.byref(probe_scale_info), on_probe_frame, None)

    g_ctx = libglib.g_main_context_default()
    wait_start = time.perf_counter()

    while not probe_done:
        while libglib.g_main_context_iteration(g_ctx, 0):
            pass

        if stop_event.is_set() or os.getppid() != parent_pid:
            libobs.obs_remove_raw_video_callback(on_probe_frame, None)
            conn.send({"status": "error", "message": "Capture stopped before screen selection completed"})
            os._exit(0)

        if portal_error is not None:
            libobs.obs_remove_raw_video_callback(on_probe_frame, None)
            conn.send({"status": "error", "message": f"Wayland portal selection failed: {portal_error}"})
            os._exit(1)

        if time.perf_counter() - wait_start > 120.0:
            libobs.obs_remove_raw_video_callback(on_probe_frame, None)
            conn.send({"status": "error", "message": "Timed out waiting for Wayland screen/window selection"})
            os._exit(1)

        time.sleep(0.005)

    libobs.obs_remove_raw_video_callback(on_probe_frame, None)

    raw_w = int(libobs.obs_source_get_width(source))
    raw_h = int(libobs.obs_source_get_height(source))
    if raw_w <= 0 or raw_h <= 0:
        conn.send({"status": "error", "message": f"Invalid source dimensions after capture start: {raw_w}x{raw_h}"})
        os._exit(1)

    # libobs rounds output_width down to a multiple of 4 and output_height down to a multiple of 2.
    cap_w = max(raw_w & ~3, 4)
    cap_h = max(raw_h & ~1, 2)

    if cap_w != probe_w or cap_h != probe_h or capture_fps != probe_fps:
        ovi.fps_num = capture_fps
        ovi.base_width = cap_w
        ovi.base_height = cap_h
        ovi.output_width = cap_w
        ovi.output_height = cap_h
        res = libobs.obs_reset_video(ctypes.byref(ovi))
        if res != 0:
            conn.send({"status": "error", "message": f"obs_reset_video({cap_w}x{cap_h}) failed (code={res})"})
            os._exit(1)

    conn.send({"status": "ok", "width": cap_w, "height": cap_h})
    try:
        reply = conn.recv()
    except EOFError:
        os._exit(0)

    if not isinstance(reply, dict) or reply.get("status") != "ok":
        os._exit(0)

    shm_name = reply["shm_name"]
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
    except Exception as e:
        conn.send({"status": "error", "message": f"Failed to attach shared memory: {e}"})
        os._exit(1)

    shm_frame: np.ndarray = np.ndarray((cap_h, cap_w, 4), dtype=np.uint8, buffer=shm.buf)
    first_frame_received = False

    @RAW_VIDEO_CALLBACK
    def on_frame(param: int, frame_ptr: Any) -> None:
        nonlocal first_frame_received
        if stop_event.is_set() or not frame_ptr:
            return
        frame = frame_ptr.contents
        linesize = frame.linesize[0]
        data_ptr = frame.data[0]
        if not data_ptr or linesize < cap_w * 4:
            return

        arr = np.ctypeslib.as_array(data_ptr, shape=(cap_h, linesize))
        bgra = arr[:, : cap_w * 4].reshape(cap_h, cap_w, 4)

        if not first_frame_received:
            if not np.any(bgra > 0):
                return
            first_frame_received = True

        with frame_lock:
            shm_frame[:] = bgra
            frame_event.set()

    scale_info = video_scale_info(
        format=VIDEO_FORMAT_BGRA,
        width=cap_w,
        height=cap_h,
        range=VIDEO_RANGE_FULL,
        colorspace=VIDEO_CS_DEFAULT,
    )
    libobs.obs_add_raw_video_callback(ctypes.byref(scale_info), on_frame, None)

    try:
        while not stop_event.is_set() and os.getppid() == parent_pid:
            while libglib.g_main_context_iteration(g_ctx, 0):
                pass
            time.sleep(0.005)
    finally:
        libobs.obs_remove_raw_video_callback(on_frame, None)
        libobs.obs_set_output_source(0, None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
        # Do not release pipewire-screen-capture-source or call obs_shutdown because
        # OBS's linux-pipewire plugin calls pw_thread_loop_wait() without locking in
        # teardown_pipewire(), causing a segfault in libpipewire.
        os._exit(0)


class ScreenshotProcessOBS(threading.Thread):
    def __init__(
        self,
        fps: int,
        frame_width: int | None,
        frame_height: int | None,
        monitor_index: int = 0,
        window_name: str | None = None,
        device: torch.device | None = None,
        crop_top: int = 0,
        crop_left: int = 0,
        crop_right: int = 0,
        crop_bottom: int = 0,
        draw_cursor_enabled: bool = True,
        **_ignore_unsupported_kwargs: object,
    ) -> None:
        super().__init__(daemon=True)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.device = device if device is not None else torch.device("cpu")
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.draw_cursor_enabled = draw_cursor_enabled

        self.cap_width = 0
        self.cap_height = 0
        self.screen_width = 0
        self.screen_height = 0

        self.frame: torch.Tensor | None = None
        self.frame_set_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cleanup_lock = threading.Lock()
        self.fps_counter: deque[float] = deque(maxlen=120)

        self.process: BaseProcess | None = None
        self.process_conn: Connection | None = None
        self.process_frame_buffer: shared_memory.SharedMemory | None = None
        self.process_stop_event: MpEvent | None = None
        self.process_frame_event: MpEvent | None = None
        self.process_frame_lock: MpLock | None = None

        if self.device.type == "cuda":
            self.cuda_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self.device)
        else:
            self.cuda_stream = None

    def set_frame_size(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

    def init_capture(self, stop_event: threading.Event | None = None) -> tuple[int, int]:
        lib_paths, reason = _resolve_obs_paths()
        if lib_paths is None or reason is not None:
            raise RuntimeError(reason or "libobs PipeWire capture is not supported")

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.process_conn = parent_conn
        self.process_stop_event = ctx.Event()
        self.process_frame_event = ctx.Event()
        self.process_frame_lock = ctx.Lock()

        proc = ctx.Process(
            target=_obs_capture_worker,
            args=(
                child_conn,
                lib_paths,
                self.fps,
                self.draw_cursor_enabled,
                os.getpid(),
                self.process_frame_lock,
                self.process_frame_event,
                self.process_stop_event,
            ),
            daemon=True,
        )
        self.process = proc
        proc.start()
        child_conn.close()

        try:
            print(
                "Waiting for Wayland screen/window selection (please select a target in the portal dialog)...",
                file=sys.stderr,
            )
            while not parent_conn.poll(0.05):
                if (stop_event is not None and stop_event.is_set()) or self.stop_event.is_set():
                    raise RuntimeError("Screen capture selection was cancelled")
                if not proc.is_alive():
                    raise RuntimeError("OBS capture worker exited unexpectedly during initialization")

            msg = parent_conn.recv()
            if not isinstance(msg, dict) or msg.get("status") != "ok":
                error_msg = msg.get("message", "Unknown OBS worker error") if isinstance(msg, dict) else str(msg)
                raise RuntimeError(error_msg)

            self.cap_width = int(msg["width"])
            self.cap_height = int(msg["height"])

            screen_width = self.cap_width - self.crop_left - self.crop_right
            screen_height = self.cap_height - self.crop_top - self.crop_bottom
            if screen_width <= 0 or screen_height <= 0:
                raise RuntimeError(
                    f"Invalid cropped size ({screen_width}x{screen_height}) from capture size "
                    f"({self.cap_width}x{self.cap_height})"
                )

            self.screen_width = screen_width
            self.screen_height = screen_height

            shm_size = self.cap_height * self.cap_width * 4
            self.process_frame_buffer = shared_memory.SharedMemory(create=True, size=shm_size)
            parent_conn.send({"status": "ok", "shm_name": self.process_frame_buffer.name})
            return self.screen_width, self.screen_height
        except Exception:
            self._cleanup_process()
            raise

    def _cleanup_process(self) -> None:
        with self.cleanup_lock:
            if self.process_stop_event is not None:
                self.process_stop_event.set()
            if self.process_conn is not None:
                try:
                    self.process_conn.close()
                except Exception:
                    pass
                self.process_conn = None
            if self.process is not None:
                self.process.join(timeout=4)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                self.process = None
            if self.process_frame_buffer is not None:
                try:
                    self.process_frame_buffer.close()
                except Exception:
                    pass
                try:
                    self.process_frame_buffer.unlink()
                except Exception:
                    pass
                self.process_frame_buffer = None

    def run(self) -> None:
        if self.process is None or self.process_frame_buffer is None:
            self.init_capture()
        if self.frame_width is None or self.frame_height is None:
            self.frame_width = self.screen_width
            self.frame_height = self.screen_height

        assert self.process is not None
        assert self.process_frame_buffer is not None
        assert self.process_frame_event is not None
        assert self.process_frame_lock is not None

        frame_buffer = torch.empty((self.screen_height, self.screen_width, 4), dtype=torch.uint8)
        shm_array: np.ndarray = np.ndarray(
            (self.cap_height, self.cap_width, 4), dtype=np.uint8, buffer=self.process_frame_buffer.buf
        )
        shm_tensor = torch.from_numpy(shm_array)
        if self.crop_top > 0 or self.crop_left > 0 or self.crop_right > 0 or self.crop_bottom > 0:
            top = self.crop_top
            bottom = self.cap_height - self.crop_bottom if self.crop_bottom > 0 else self.cap_height
            left = self.crop_left
            right = self.cap_width - self.crop_right if self.crop_right > 0 else self.cap_width
            shm_view = shm_tensor[top:bottom, left:right, :]
        else:
            shm_view = shm_tensor

        try:
            while not self.stop_event.is_set():
                tick = time.perf_counter()
                while not self.process_frame_event.wait(0.1):
                    if self.stop_event.is_set():
                        return
                    if not self.process.is_alive():
                        raise RuntimeError("OBS capture process is already dead")

                with self.process_frame_lock:
                    frame_buffer.copy_(shm_view)
                    self.process_frame_event.clear()

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        if frame.shape[1:] != (self.frame_height, self.frame_width):
                            frame = TF.resize(
                                frame,
                                size=[self.frame_height, self.frame_width],
                                interpolation=InterpolationMode.BILINEAR,
                                antialias=True,
                            )
                    frame.record_stream(self.cuda_stream)
                else:
                    frame = frame_buffer.to(self.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    if frame.shape[1:] != (self.frame_height, self.frame_width):
                        frame = TF.resize(
                            frame,
                            size=[self.frame_height, self.frame_width],
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        )

                with self.frame_lock:
                    self.frame = frame
                    self.frame_set_event.set()

                process_time = time.perf_counter() - tick
                with self.fps_lock:
                    self.fps_counter.append(process_time)
        finally:
            with self.frame_lock:
                self.frame = None
            self.stop_event.set()
            self.frame_set_event.set()
            self._cleanup_process()

    def get_frame(self) -> torch.Tensor:
        while not self.frame_set_event.wait(1):
            if self.stop_event.is_set() or not self.is_alive():
                raise RuntimeError("OBS screenshot thread is already dead")
        if self.stop_event.is_set():
            raise RuntimeError("OBS screenshot thread is already dead")
        with self.frame_lock:
            if self.cuda_stream is not None:
                torch.cuda.current_stream().wait_stream(self.cuda_stream)
            frame = self.frame
            self.frame_set_event.clear()
        if frame is None:
            raise RuntimeError("OBS screenshot thread is already dead")
        return frame

    def get_fps(self) -> float:
        with self.fps_lock:
            if self.fps_counter:
                mean_processing_time = sum(self.fps_counter) / len(self.fps_counter)
                return 1.0 / mean_processing_time
            else:
                return 0.0

    def stop(self) -> None:
        self.stop_event.set()
        if self.process_stop_event is not None:
            self.process_stop_event.set()
        if self.ident is not None:
            self.join(timeout=4)
        self._cleanup_process()
