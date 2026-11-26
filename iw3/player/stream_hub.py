from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

FrameData = tuple[bytes, float]
FrameProvider = FrameData | Callable[[], FrameData]
STREAM_URI = "/api/stream"
STREAM_WS_URI = "/api/stream/ws"


@dataclass(slots=True)
class EncodedPacket:
    data: bytes
    timestamp_us: int
    is_keyframe: bool
    codec_string: str
    codec_family: str
    video_codec: str
    width: int
    height: int
    epoch: int = 0


@dataclass(slots=True)
class WsSubscriber:
    client_id: int
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[EncodedPacket | dict[str, Any] | None]
    requested_codecs: list[str] = field(default_factory=lambda: ["hevc", "h264"])
    paused: bool = False
    waiting_for_keyframe: bool = True


class StreamFrameHub:
    """Thread-safe MJPEG and encoded video frame hub bridging producer threads and async clients."""

    def __init__(
        self,
        lock: threading.Lock | None = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        fps: int = 30,
        stereo_format: str = "SBS_HALF",
        stream_content_type: str = "image/jpeg",
        video_codec: str = "libx264",
        available_stream_codecs: list[str] | None = None,
    ) -> None:
        self.lock = lock if lock is not None else threading.Lock()
        self.op_lock = threading.Lock()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.stereo_format = stereo_format
        self.stream_uri = STREAM_URI
        self.ws_uri = STREAM_WS_URI
        self.stream_content_type = stream_content_type
        self.video_codec = video_codec
        self.available_stream_codecs: list[str] = (
            list(available_stream_codecs) if available_stream_codecs is not None else ["h264"]
        )
        self.active_video_codec: str | None = None

        self.frame_data: FrameData | None = None
        self.frame_data_raw: FrameProvider | None = None
        self.stream_token = f"{random.getrandbits(64):016x}"
        self.shutdown_event = threading.Event()
        self.fps_counter: deque[tuple[int, float]] = deque(maxlen=120)
        self.active_clients: int = 0
        self._ws_subscribers: dict[int, WsSubscriber] = {}
        self._keyframe_requested: bool = False
        self.last_activity_time: float = time.perf_counter()
        self.service: Any = None
        self._stream_epoch: int = 0
        self.last_error: str | None = None

    def reset_token(self) -> None:
        self.stream_token = f"{random.getrandbits(64):016x}"

    def touch(self) -> None:
        with self.op_lock:
            self.last_activity_time = time.perf_counter()

    def clear_frames(self, stop_clients: bool = False, error_message: str | None = None) -> None:
        with self.lock:
            self.frame_data = None
            self.frame_data_raw = None
        subscribers: list[WsSubscriber] = []
        with self.op_lock:
            self.fps_counter.clear()
            self.active_video_codec = None
            if error_message is not None:
                self.last_error = error_message
            elif not stop_clients:
                self.last_error = None
            if stop_clients:
                self._stream_epoch += 1
                subscribers = list(self._ws_subscribers.values())
        if stop_clients:
            signal = {"type": "error", "message": error_message} if error_message is not None else None
            for sub in subscribers:
                self._notify_subscriber_close(sub, signal=signal)

    def _notify_subscriber_close(self, sub: WsSubscriber, signal: dict[str, Any] | None = None) -> None:
        def _put_close() -> None:
            while not sub.queue.empty():
                try:
                    sub.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                if signal is not None:
                    sub.queue.put_nowait(signal)
                sub.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        try:
            sub.loop.call_soon_threadsafe(_put_close)
        except RuntimeError:
            pass

    def fail_ws_streams(self, reason: str = "encoder_error") -> None:
        with self.lock:
            self.frame_data = None
            self.frame_data_raw = None
        subscribers: list[WsSubscriber] = []
        with self.op_lock:
            self.fps_counter.clear()
            self.active_video_codec = None
            self.available_stream_codecs = []
            self._stream_epoch += 1
            subscribers = list(self._ws_subscribers.values())
        signal = {"type": "fallback", "reason": reason}
        for sub in subscribers:
            self._notify_subscriber_close(sub, signal=signal)

    def update_stream_params(
        self,
        frame_width: int,
        frame_height: int,
        fps: int,
        stereo_format: str,
        video_codec: str | None = None,
        available_stream_codecs: list[str] | None = None,
    ) -> None:
        with self.op_lock:
            self.frame_width = frame_width
            self.frame_height = frame_height
            self.fps = fps
            self.stereo_format = stereo_format
            if video_codec is not None:
                self.video_codec = video_codec
            if available_stream_codecs is not None:
                self.available_stream_codecs = list(available_stream_codecs)
            self.last_activity_time = time.perf_counter()
            self.last_error = None
            self.reset_token()

    @property
    def active_ws_clients(self) -> int:
        with self.op_lock:
            return len(self._ws_subscribers)

    def has_mjpeg_clients(self) -> bool:
        with self.op_lock:
            return self.active_clients > 0

    def has_ws_clients(self) -> bool:
        with self.op_lock:
            return any(not sub.paused for sub in self._ws_subscribers.values())

    def has_ws_subscribers(self) -> bool:
        with self.op_lock:
            return len(self._ws_subscribers) > 0

    def has_any_clients(self) -> bool:
        with self.op_lock:
            return self.active_clients > 0 or len(self._ws_subscribers) > 0

    @staticmethod
    def _intersect_requested_codecs(subs: list[WsSubscriber]) -> list[str]:
        common = list(subs[0].requested_codecs)
        for sub in subs[1:]:
            allowed = set(sub.requested_codecs)
            common = [c for c in common if c in allowed]
        return common

    def begin_frame_encode(self) -> tuple[bool, bool, list[str] | None, bool, bool]:
        with self.op_lock:
            has_ws_subs = bool(self.available_stream_codecs) and len(self._ws_subscribers) > 0
            active_subs = [sub for sub in self._ws_subscribers.values() if not sub.paused]
            has_active_ws = has_ws_subs and len(active_subs) > 0
            common: list[str] | None = None
            force_keyframe = False
            if has_active_ws:
                common = self._intersect_requested_codecs(active_subs)
                force_keyframe = self._keyframe_requested
                self._keyframe_requested = False
            elif has_ws_subs:
                common = self._intersect_requested_codecs(list(self._ws_subscribers.values()))
            has_mjpeg = self.active_clients > 0
        return has_ws_subs, has_active_ws, common, force_keyframe, has_mjpeg

    def request_keyframe(self) -> None:
        with self.op_lock:
            self._keyframe_requested = True

    def consume_keyframe_request(self) -> bool:
        with self.op_lock:
            requested = self._keyframe_requested
            self._keyframe_requested = False
            return requested

    def get_ws_requested_codecs(self) -> list[str] | None:
        with self.op_lock:
            active_subs = [sub for sub in self._ws_subscribers.values() if not sub.paused]
            if not active_subs:
                active_subs = list(self._ws_subscribers.values())
            if not active_subs:
                return None
            return self._intersect_requested_codecs(active_subs)

    def subscribe_ws(
        self,
        requested_codecs: list[str] | None = None,
        queue_size: int = 8,
    ) -> tuple[WsSubscriber, int]:
        loop = asyncio.get_running_loop()
        codecs = [c.lower() for c in requested_codecs if c] if requested_codecs else ["hevc", "h264"]
        sub = WsSubscriber(
            client_id=random.getrandbits(64),
            loop=loop,
            queue=asyncio.Queue(maxsize=max(2, queue_size)),
            requested_codecs=codecs,
            paused=False,
            waiting_for_keyframe=True,
        )
        with self.op_lock:
            self._ws_subscribers[sub.client_id] = sub
            self._keyframe_requested = True
            self.last_activity_time = time.perf_counter()
            epoch = self._stream_epoch
        return sub, epoch

    def unsubscribe_ws(self, client_id: int) -> None:
        with self.op_lock:
            self._ws_subscribers.pop(client_id, None)
            self.last_activity_time = time.perf_counter()

    def set_ws_paused(self, client_id: int, paused: bool) -> None:
        with self.op_lock:
            sub = self._ws_subscribers.get(client_id)
            if sub is not None:
                was_paused = sub.paused
                sub.paused = paused
                if not paused and was_paused:
                    sub.waiting_for_keyframe = True
                    self._keyframe_requested = True
            self.last_activity_time = time.perf_counter()

    def _deliver_ws_packet(self, sub: WsSubscriber, packet: EncodedPacket) -> None:
        if sub.paused:
            return
        if sub.waiting_for_keyframe:
            if not packet.is_keyframe:
                return
            sub.waiting_for_keyframe = False

        if sub.queue.full():
            # Drop stale buffered packets and force a clean keyframe resync on the client.
            while not sub.queue.empty():
                try:
                    sub.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if not packet.is_keyframe:
                sub.waiting_for_keyframe = True
                self.request_keyframe()
                return

        try:
            sub.queue.put_nowait(packet)
        except asyncio.QueueFull:
            if not packet.is_keyframe:
                sub.waiting_for_keyframe = True
                self.request_keyframe()

    def record_ws_frame_sent(self, client_id: int) -> None:
        now = time.perf_counter()
        with self.op_lock:
            self.fps_counter.append((client_id, now))

    def push_encoded_packet(self, packet: EncodedPacket) -> None:
        now = time.perf_counter()
        with self.op_lock:
            epoch = self._stream_epoch
            self.active_video_codec = packet.video_codec
            self.last_activity_time = now
            subscribers = list(self._ws_subscribers.values())

        stamped_packet = (
            packet
            if packet.epoch == epoch
            else EncodedPacket(
                data=packet.data,
                timestamp_us=packet.timestamp_us,
                is_keyframe=packet.is_keyframe,
                codec_string=packet.codec_string,
                codec_family=packet.codec_family,
                video_codec=packet.video_codec,
                width=packet.width,
                height=packet.height,
                epoch=epoch,
            )
        )
        for sub in subscribers:
            if sub.paused:
                continue
            try:
                sub.loop.call_soon_threadsafe(self._deliver_ws_packet, sub, stamped_packet)
            except RuntimeError:
                pass

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
                "stream_token": self.stream_token,
                "stream_uri": self.stream_uri,
                "ws_uri": self.ws_uri,
                "video_codec": self.video_codec,
                "available_stream_codecs": list(self.available_stream_codecs),
                "active_video_codec": self.active_video_codec,
                "error": self.last_error,
            }
        active = bool(service.is_running()) if service is not None and hasattr(service, "is_running") else True
        info["active"] = active
        buffer_ms = 0
        if service is not None and hasattr(service, "get_settings"):
            settings = service.get_settings()
            if isinstance(settings, dict):
                buffer_ms = int(settings.get("buffer_ms", 0))
        info["buffer_ms"] = buffer_ms
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
