from __future__ import annotations

import atexit
import os
from os import path
import shutil
import subprocess
import sys
import threading
import time

import psutil
HEARTBEAT_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 10.0
GUARD_POLL_INTERVAL = 1.0


class DisplayModeUnavailable(RuntimeError):
    pass


def _print_error(message):
    print(message, file=sys.stderr)


def _supports_linux_display_mode():
    if sys.platform != "linux":
        return True

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        return False

    if not os.environ.get("DISPLAY"):
        return False

    return shutil.which("xrandr") is not None


def get_pymonctl():
    if sys.platform not in {"win32", "linux"}:
        raise DisplayModeUnavailable(f"unsupported platform: {sys.platform}")

    if sys.platform == "linux" and not _supports_linux_display_mode():
        raise DisplayModeUnavailable(
            "display mode switching requires X11 with xrandr on Linux"
        )

    try:
        import pymonctl as pmc
    except SystemExit as exc:
        raise DisplayModeUnavailable(
            "PyMonCtl exited during import; Linux mode switching requires X11/xrandr"
        ) from exc
    except Exception as exc:
        raise DisplayModeUnavailable(f"PyMonCtl is not available: {exc}") from exc

    return pmc


def _mode_frequency(mode):
    return float(getattr(mode, "frequency", 0.0) or 0.0)


def _serialize_mode(mode):
    return {
        "width": int(mode.width),
        "height": int(mode.height),
        "frequency": float(_mode_frequency(mode)),
    }


def _mode_equals(mode_a, mode_b):
    return (
        int(mode_a.width) == int(mode_b.width) and
        int(mode_a.height) == int(mode_b.height) and
        abs(_mode_frequency(mode_a) - _mode_frequency(mode_b)) < 0.01
    )


def _same_aspect(mode, width, height):
    return int(mode.width) * int(height) == int(mode.height) * int(width)


def _find_mode_by_size(modes, width, height, preferred_frequency):
    candidates = [
        mode for mode in modes
        if int(mode.width) == int(width) and int(mode.height) == int(height)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda mode: abs(_mode_frequency(mode) - float(preferred_frequency or 0.0))
    )


def find_restore_mode(modes, original_mode):
    match = _find_mode_by_size(
        modes,
        original_mode["width"],
        original_mode["height"],
        original_mode.get("frequency", 0.0),
    )
    if match is not None:
        return match
    return None


def find_doubled_mode(original_mode, modes):
    desired_width = int(original_mode.width) * 2
    desired_height = int(original_mode.height) * 2
    current_width = int(original_mode.width)
    current_height = int(original_mode.height)
    preferred_frequency = _mode_frequency(original_mode)

    exact = _find_mode_by_size(modes, desired_width, desired_height, preferred_frequency)
    if exact is not None:
        return exact

    target_area = desired_width * desired_height

    larger_modes = [
        mode for mode in modes
        if int(mode.width) >= current_width and int(mode.height) >= current_height
    ]
    if not larger_modes:
        return None

    same_aspect = [
        mode for mode in larger_modes
        if _same_aspect(mode, desired_width, desired_height)
    ]

    candidates = same_aspect if same_aspect else larger_modes
    if not candidates:
        return None

    def sort_key(mode):
        width = int(mode.width)
        height = int(mode.height)
        area = width * height
        below_target = int(width < desired_width or height < desired_height)
        area_distance = abs(area - target_area)
        freq_distance = abs(_mode_frequency(mode) - preferred_frequency)
        return (below_target, area_distance, freq_distance, -area)

    best = min(candidates, key=sort_key)
    if int(best.width) == current_width and int(best.height) == current_height:
        return None
    return best


def _window_center(window):
    rect = window.GetScreenRect()
    return {
        "x": int(rect.x + (rect.width // 2)),
        "y": int(rect.y + (rect.height // 2)),
    }


def _get_monitor_for_window(pmc, window):
    center = _window_center(window)
    monitors = pmc.findMonitorsAtPoint(center["x"], center["y"])
    if not monitors:
        raise RuntimeError("monitor for Local Viewer window was not found")

    monitor = monitors[0]
    return monitor, center
def _current_process_create_time():
    try:
        return psutil.Process(os.getpid()).create_time()
    except psutil.Error:
        return None


def _process_matches(pid, create_time):
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running():
            return False
        if create_time is None:
            return True
        return abs(proc.create_time() - float(create_time)) < 1.0
    except (psutil.Error, TypeError, ValueError):
        return False


def heartbeat_expired(last_heartbeat, now=None):
    if now is None:
        now = time.time()

    try:
        heartbeat = float(last_heartbeat or 0.0)
    except (TypeError, ValueError):
        return True
    return (now - heartbeat) > HEARTBEAT_TIMEOUT


def _find_monitor_from_state(pmc, state):
    monitor_name = state.get("monitor_name")
    if monitor_name:
        monitor = pmc.findMonitorWithName(monitor_name)
        if monitor is not None:
            return monitor

    center = state.get("monitor_center") or {}
    x = center.get("x")
    y = center.get("y")
    if x is None or y is None:
        return None

    monitors = pmc.findMonitorsAtPoint(int(x), int(y))
    if not monitors:
        return None
    return monitors[0]


def restore_original_mode(state, reason="restore"):
    pmc = get_pymonctl()
    monitor = _find_monitor_from_state(pmc, state)
    if monitor is None:
        raise RuntimeError("original monitor could not be found for restore")

    original_mode = state["original_mode"]
    restore_mode = find_restore_mode(monitor.allModes, original_mode)
    if restore_mode is None:
        restore_mode = pmc.DisplayMode(
            int(original_mode["width"]),
            int(original_mode["height"]),
            float(original_mode.get("frequency", 0.0)),
        )

    current_mode = monitor.mode
    if current_mode is not None and _mode_equals(current_mode, restore_mode):
        return False

    monitor.setMode(restore_mode)
    _print_error(
        f"Restored Local Viewer display mode on {monitor.name} ({reason})"
    )
    return True


def _spawn_guard(state):
    command = [
        sys.executable,
        "-m",
        "iw3.desktop.display_mode_guard",
        "--parent-pid",
        str(state["parent_pid"]),
        "--parent-create-time",
        str(state["parent_create_time"] if state["parent_create_time"] is not None else -1.0),
        "--monitor-name",
        str(state["monitor_name"]),
        "--monitor-center-x",
        str(state["monitor_center"]["x"]),
        "--monitor-center-y",
        str(state["monitor_center"]["y"]),
        "--width",
        str(state["original_mode"]["width"]),
        "--height",
        str(state["original_mode"]["height"]),
        "--frequency",
        str(state["original_mode"]["frequency"]),
    ]

    kwargs = {
        "args": command,
        "cwd": path.join(path.dirname(__file__), "..", ".."),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.PIPE,
        "close_fds": True,
        "text": True,
        "bufsize": 1,
    }

    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS |
            subprocess.CREATE_NEW_PROCESS_GROUP |
            subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(**kwargs)


class LocalViewerDisplayModeController:
    def __init__(self):
        self.lock = threading.RLock()
        self.active_state = None
        self.guard_process = None
        self.heartbeat_stop_event = None
        self.heartbeat_thread = None
        atexit.register(self.close)

    def is_active(self):
        with self.lock:
            return self.active_state is not None

    def toggle_for_window(self, window):
        if self.is_active():
            return self.restore()

        if not window.IsFullScreen():
            return False

        return self.activate_for_window(window)

    def activate_for_window(self, window):
        with self.lock:
            if self.active_state is not None:
                return False

            pmc = get_pymonctl()
            monitor, center = _get_monitor_for_window(pmc, window)
            original_mode = monitor.mode
            if original_mode is None:
                raise RuntimeError("current monitor mode could not be determined")

            target_mode = find_doubled_mode(original_mode, monitor.allModes)
            if target_mode is None:
                raise RuntimeError(
                    f"no larger display mode is available for {monitor.name} "
                    f"from {original_mode.width}x{original_mode.height}"
                )

            state = {
                "parent_pid": os.getpid(),
                "parent_create_time": _current_process_create_time(),
                "monitor_name": monitor.name,
                "monitor_center": center,
                "original_mode": _serialize_mode(original_mode),
            }

            self.guard_process = _spawn_guard(state)

            try:
                monitor.setMode(target_mode)
            except Exception:
                self._shutdown_guard_locked()
                raise

            self.active_state = state
            self._start_heartbeat()
            _print_error(
                f"Enabled Local Viewer 2x display mode on {monitor.name}: "
                f"{original_mode.width}x{original_mode.height} -> "
                f"{target_mode.width}x{target_mode.height}"
            )
            return True

    def restore(self):
        with self.lock:
            if self.active_state is None:
                return False
            state = dict(self.active_state)

        restore_original_mode(state, reason="toggle off")

        with self.lock:
            self.active_state = None
            self._stop_heartbeat_locked()
            self._shutdown_guard_locked()
        return True

    def close(self):
        with self.lock:
            if self.active_state is None:
                self._stop_heartbeat_locked()
                self._shutdown_guard_locked()
                return False
        return self.restore()

    def _start_heartbeat(self):
        self._stop_heartbeat_locked()
        self.heartbeat_stop_event = threading.Event()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat_locked(self):
        if self.heartbeat_stop_event is not None:
            self.heartbeat_stop_event.set()
        heartbeat_thread = self.heartbeat_thread
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2.0)
        self.heartbeat_stop_event = None
        self.heartbeat_thread = None

    def _shutdown_guard_locked(self):
        guard_process = self.guard_process
        self.guard_process = None
        if guard_process is None:
            return

        if guard_process.stdin is not None:
            try:
                guard_process.stdin.write("stop\n")
                guard_process.stdin.flush()
            except Exception:
                pass

        try:
            guard_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                guard_process.terminate()
                guard_process.wait(timeout=2.0)
            except Exception:
                try:
                    guard_process.kill()
                except Exception:
                    pass
        finally:
            if guard_process.stdin is not None:
                try:
                    guard_process.stdin.close()
                except Exception:
                    pass

    def _heartbeat_loop(self):
        while self.heartbeat_stop_event is not None and not self.heartbeat_stop_event.wait(HEARTBEAT_INTERVAL):
            with self.lock:
                if self.active_state is None or self.guard_process is None:
                    return
                try:
                    if self.guard_process.stdin is None:
                        return
                    self.guard_process.stdin.write("ping\n")
                    self.guard_process.stdin.flush()
                except Exception as exc:
                    _print_error(f"failed to write display mode heartbeat: {exc}")
                    return


def guard_main(state):
    stop_event = threading.Event()
    pipe_closed_event = threading.Event()
    last_heartbeat = {"value": time.time()}

    def stdin_loop():
        try:
            while True:
                line = sys.stdin.readline()
                if line == "":
                    pipe_closed_event.set()
                    return
                command = line.strip().lower()
                if command == "ping":
                    last_heartbeat["value"] = time.time()
                elif command == "stop":
                    stop_event.set()
                    return
        except Exception:
            pipe_closed_event.set()

    reader_thread = threading.Thread(target=stdin_loop, daemon=True)
    reader_thread.start()

    while True:
        if stop_event.is_set():
            return 0

        if not _process_matches(state["parent_pid"], state["parent_create_time"]):
            try:
                restore_original_mode(state, reason="guard recovery")
            except Exception:
                pass
            return 0

        if pipe_closed_event.is_set() or heartbeat_expired(last_heartbeat["value"]):
            try:
                restore_original_mode(state, reason="guard recovery")
            except Exception:
                pass
            return 0

        time.sleep(GUARD_POLL_INTERVAL)
