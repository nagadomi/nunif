from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import defaultdict, deque
from typing import Any, AsyncGenerator, Callable

FrameData = tuple[bytes, float]
FrameProvider = FrameData | Callable[[], FrameData]


class StreamFrameHub:
    """Thread-safe MJPEG frame hub bridging producer threads and async HTTP clients."""

    def __init__(
        self,
        lock: threading.Lock | None = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        fps: int = 30,
        stereo_format: str = "SBS_HALF",
        stream_uri: str = "/stream.jpg",
        stream_content_type: str = "image/jpeg",
    ) -> None:
        self.lock = lock if lock is not None else threading.Lock()
        self.op_lock = threading.Lock()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.stereo_format = stereo_format
        self.stream_uri = stream_uri
        self.stream_content_type = stream_content_type

        self.frame_data: FrameData | None = None
        self.frame_data_raw: FrameProvider | None = None
        self.process_token = f"{random.getrandbits(64):016x}"
        self.shutdown_event = threading.Event()
        self.fps_counter: deque[tuple[int, float]] = deque(maxlen=120)
        self.active_clients: int = 0
        self.last_activity_time: float = time.perf_counter()
        self.service: Any = None
        self._stream_epoch: int = 0

    def reset_token(self) -> None:
        self.process_token = f"{random.getrandbits(64):016x}"

    def touch(self) -> None:
        with self.op_lock:
            self.last_activity_time = time.perf_counter()

    def clear_frames(self, stop_clients: bool = False) -> None:
        with self.lock:
            self.frame_data = None
            self.frame_data_raw = None
        with self.op_lock:
            self.fps_counter.clear()
            if stop_clients:
                self._stream_epoch += 1

    def update_stream_params(
        self,
        frame_width: int,
        frame_height: int,
        fps: int,
        stereo_format: str,
    ) -> None:
        with self.op_lock:
            self.frame_width = frame_width
            self.frame_height = frame_height
            self.fps = fps
            self.stereo_format = stereo_format
            self.last_activity_time = time.perf_counter()
            self.reset_token()

    def set_frame_data(self, frame_data: FrameProvider) -> None:
        with self.lock:
            self.frame_data_raw = frame_data

    def get_frame_data(self) -> FrameData | None:
        with self.lock:
            frame_data = self.frame_data_raw
            self.frame_data_raw = None
            if callable(frame_data):
                self.frame_data = frame_data()
            elif frame_data is not None:
                self.frame_data = frame_data
            return self.frame_data

    def get_fps(self) -> float:
        now = time.perf_counter()
        tid_times: dict[int, list[float]] = defaultdict(list)
        with self.op_lock:
            for tid, t in self.fps_counter:
                if now - t <= 2.0:
                    tid_times[tid].append(t)

        fps_list: list[float] = []
        for times in tid_times.values():
            if len(times) < 2:
                continue
            diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
            mean_diff = sum(diffs) / len(diffs)
            if mean_diff > 0:
                fps_list.append(1.0 / mean_diff)

        if fps_list:
            return sum(fps_list) / len(fps_list)
        return 0.0

    def get_stream_info(self) -> dict[str, Any]:
        with self.op_lock:
            self.last_activity_time = time.perf_counter()
            service = self.service
            info: dict[str, Any] = {
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": self.fps,
                "stereo_format": self.stereo_format,
                "process_token": self.process_token,
                "stream_uri": self.stream_uri,
            }
        active = bool(service.is_running()) if service is not None and hasattr(service, "is_running") else True
        info["active"] = active
        return info

    async def async_stream_generator(self) -> AsyncGenerator[bytes, None]:
        generator_id = random.getrandbits(64)
        data_tick = 0.0
        content_type_bytes = self.stream_content_type.encode("ascii")

        with self.op_lock:
            epoch = self._stream_epoch
            self.active_clients += 1
            self.last_activity_time = time.perf_counter()

        try:
            while not self.shutdown_event.is_set() and self._stream_epoch == epoch:
                with self.lock:
                    has_callable = callable(self.frame_data_raw)

                if has_callable:
                    data = await asyncio.to_thread(self.get_frame_data)
                else:
                    data = self.get_frame_data()

                if data is not None:
                    frame, tick = data
                    if tick > data_tick:
                        data_tick = tick
                        now = time.perf_counter()
                        with self.op_lock:
                            self.fps_counter.append((generator_id, now))
                            self.last_activity_time = now
                        header = (
                            b"--frame\r\nContent-Type: "
                            + content_type_bytes
                            + f"\r\nContent-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        )
                        yield header + frame + b"\r\n"
                        continue

                await asyncio.sleep(0.002)
        finally:
            with self.op_lock:
                self.active_clients = max(0, self.active_clients - 1)
                self.last_activity_time = time.perf_counter()
