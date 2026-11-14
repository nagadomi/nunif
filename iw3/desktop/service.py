from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from .utils import (
    cleanup_desktop_pipeline,
    ensure_default_stereo_format,
    get_runtime_stream_settings,
    init_desktop_pipeline,
    resolve_player_stereo_format,
    run_desktop_processing_loop,
    update_runtime_stream_settings,
    validate_screenshot_args,
)

if TYPE_CHECKING:
    from ..player.stream_hub import StreamFrameHub


class DesktopStreamService:
    """On-demand desktop capture, depth estimation, and stereo streaming service."""

    def __init__(
        self,
        args: Any,
        stream_hub: StreamFrameHub,
        idle_timeout: float = 10.0,
        init_wxapp: bool = False,
    ) -> None:
        self.args = args
        self.stream_hub = stream_hub
        self.stream_hub.service = self
        self.idle_timeout = idle_timeout
        self.init_wxapp = init_wxapp
        self._op_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._worker_stop_event: threading.Event | None = None
        self._ready_event: threading.Event | None = None
        self._error: BaseException | None = None
        ensure_default_stereo_format(self.args)
        validate_screenshot_args(self.args)

    def is_running(self) -> bool:
        with self._op_lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def ensure_running(self, timeout: float = 30.0) -> None:
        self.stream_hub.touch()
        for _ in range(2):
            with self._op_lock:
                if self._worker_thread is not None and self._worker_thread.is_alive():
                    ready_event = self._ready_event
                    worker_stop_event = self._worker_stop_event
                    started_new = False
                else:
                    self._error = None
                    self._worker_stop_event = threading.Event()
                    self._ready_event = threading.Event()
                    ready_event = self._ready_event
                    worker_stop_event = self._worker_stop_event
                    self._worker_thread = threading.Thread(
                        target=self._run_worker,
                        args=(worker_stop_event, ready_event),
                        daemon=True,
                    )
                    self._worker_thread.start()
                    started_new = True

            if ready_event is not None and not ready_event.wait(timeout=timeout):
                if worker_stop_event is not None:
                    worker_stop_event.set()
                raise TimeoutError("Timed out starting desktop stream")
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            if started_new or self.is_running():
                return

    def stop(self) -> None:
        with self._op_lock:
            thread = self._worker_thread
            stop_event = self._worker_stop_event
            if stop_event is not None:
                stop_event.set()
        self.stream_hub.clear_frames(stop_clients=True)
        if thread is not None and thread is not threading.current_thread():
            if thread.ident is not None:
                thread.join(timeout=10.0)
        with self._op_lock:
            if self._worker_thread is thread:
                self._worker_thread = None
                self._worker_stop_event = None

    def get_settings(self) -> dict[str, Any]:
        return get_runtime_stream_settings(self.args)

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        return update_runtime_stream_settings(self.args, updates)

    def _is_idle(self) -> bool:
        if self.idle_timeout <= 0:
            return False
        if self.stream_hub.active_clients > 0:
            return False
        return (time.perf_counter() - self.stream_hub.last_activity_time) > self.idle_timeout

    def _run_worker(self, stop_event: threading.Event, ready_event: threading.Event) -> None:
        pipeline = None
        depth_model = self.args.state["depth_model"]
        try:
            pipeline = init_desktop_pipeline(
                self.args,
                init_wxapp=self.init_wxapp,
                stop_event=stop_event,
            )
            self.stream_hub.clear_frames(stop_clients=False)
            self.stream_hub.update_stream_params(
                frame_width=pipeline.output_frame_width,
                frame_height=pipeline.output_frame_height,
                fps=int(self.args.stream_fps),
                stereo_format=resolve_player_stereo_format(self.args),
            )
            pipeline.screenshot_thread.start()
            self.stream_hub.touch()
            ready_event.set()
            run_desktop_processing_loop(
                self.args,
                pipeline,
                self.stream_hub,
                stop_event=stop_event,
                idle_check=self._is_idle,
            )
        except BaseException as exc:
            self._error = exc
        finally:
            ready_event.set()
            self.stream_hub.clear_frames(stop_clients=True)
            cleanup_desktop_pipeline(pipeline, depth_model)
            fps_event = self.args.state.get("fps_event")
            if fps_event is not None:
                fps_event.update(0.0, 0.0, 0.0, None)
            with self._op_lock:
                if self._worker_thread is threading.current_thread():
                    self._worker_thread = None
                    self._worker_stop_event = None
