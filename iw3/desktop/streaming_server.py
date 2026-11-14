"""Mixed-Replace(MJPEG) Streaming Server"""

import base64
import io
import json
import random
import sys
import threading
import time
from collections import defaultdict, deque
from socketserver import ThreadingMixIn
from string import Template
from typing import Any
from wsgiref.simple_server import WSGIServer, make_server

STATUS_OK = "200 OK"


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False


class StreamingServer:
    def __init__(
        self,
        port,
        lock,
        frame_width,
        frame_height,
        fps,
        index_template,
        stream_uri="/stream.jpg",
        stream_content_type="image/jpeg",
        auth=None,
        host="",
    ):
        self.port = port
        self.host = host
        self.lock = lock
        self.op_lock = threading.Lock()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.delay = 1.0 / fps
        self.index_template = index_template
        self.stream_uri = stream_uri
        self.stream_content_type = stream_content_type

        self.frame_data = None
        self.frame_data_raw = None

        self.server = None
        self.thread = None
        self.process_token = None
        self.shutdown_event = threading.Event()
        self.fps_counter = deque(maxlen=120)

        if auth is not None:
            user, password = auth
            self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        else:
            self.auth = None

    def _stop(self):
        self.shutdown_event.set()
        time.sleep(0.1)
        if self.server is not None:
            self.server.server_close()
            self.server.shutdown()
            self.server = None
        if self.thread is not None:
            if self.thread.ident is not None:
                self.thread.join()
            self.thread = None

        time.sleep(0.1)
        self.shutdown_event.clear()

    def _start(self):
        self.server = make_server(self.host, self.port, self.handle, ThreadingWSGIServer)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.process_token = "%016x" % random.getrandbits(64)

    def stop(self):
        with self.op_lock:
            self._stop()

    def start(self):
        with self.op_lock:
            self._stop()
            self._start()

    def set_frame_data(self, frame_data):
        # frame_data = (image_data, time) or
        # frame_data = callable()->(image_data, time)
        with self.lock:
            self.frame_data_raw = frame_data

    def get_frame_data(self):
        with self.lock:
            frame_data = self.frame_data_raw
            self.frame_data_raw = None  # handled
            if callable(frame_data):
                self.frame_data = frame_data()
            elif frame_data is not None:
                self.frame_data = frame_data

            return self.frame_data

    def get_fps(self):
        tid_times = defaultdict(lambda: [])
        with self.op_lock:
            for tid, t in self.fps_counter:
                tid_times[tid].append(t)

        fps = []
        for tid, times in tid_times.items():
            prev = None
            diff = []
            for t in times:
                if prev is not None:
                    diff.append(t - prev)
                prev = t
            if diff:
                fps.append(1.0 / (sum(diff) / len(diff)))
        if fps:
            return sum(fps) / len(fps)
        else:
            return 0

    def send_image_stream(self, start_response):
        def gen():
            generator_id = random.getrandbits(64)
            data_tick = 0
            frame = None
            send_at = time.perf_counter()
            bio = io.BytesIO()
            bio.write(b"--frame\r\n" + f"Content-Type: {self.stream_content_type}".encode() + b"\r\n\r\n")
            pos = bio.tell()
            while True:
                try:
                    data = self.get_frame_data()
                    if data is not None:
                        frame, tick = data
                        if tick > data_tick:
                            data_tick = tick
                            with self.op_lock:
                                self.fps_counter.append((generator_id, time.perf_counter()))
                            bio.seek(pos, io.SEEK_SET)
                            bio.truncate(pos)
                            bio.write(frame)
                            yield bio.getbuffer().tobytes()
                    if self.shutdown_event.is_set():
                        break
                    if False:  # True if needed
                        now = time.perf_counter()
                        if now - send_at < self.delay:
                            time.sleep(self.delay - (now - send_at))
                        send_at = now
                    else:
                        # busy waiting
                        time.sleep(1 / 1000)
                except GeneratorExit:
                    raise
                except:  # noqa
                    print("StreamingServer", sys.exc_info(), file=sys.stderr)
                    raise
            yield b""

        start_response(STATUS_OK, [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
        return gen()

    def send_index(self, start_response):
        template = Template(self.index_template)
        page_data = template.substitute(
            frame_width=self.frame_width, frame_height=self.frame_height, fps=self.fps, stream_uri=self.stream_uri
        ).encode()
        start_response(STATUS_OK, [("Content-type", "text/html; charset=utf-8")])
        return [page_data]

    def send_404(self, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"404 Not Found"]

    def send_process_token(self, start_response):
        start_response(STATUS_OK, [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"token": self.process_token}).encode()]

    def handle(self, environ, start_response):
        uri = environ["PATH_INFO"]
        #  print("request", uri)

        if self.auth is not None:
            # HTTP Basic Authentication
            auth = environ.get("HTTP_AUTHORIZATION", "")
            #  print("auth", auth)
            if auth != self.auth:
                start_response(
                    "401 Unauthorized",
                    [("WWW-Authenticate", "Basic charset=utf-8"), ("Content-Type", "text/plain; charset=utf-8")],
                )
                return [b"Authorization Required"]

        if uri == "/":
            return self.send_index(start_response)
        elif uri == "/process_token":
            return self.send_process_token(start_response)
        elif uri == self.stream_uri:
            return self.send_image_stream(start_response)
        else:
            return self.send_404(start_response)


class WebXRStreamingServer:
    """HTTPS WebXR streaming server backed by iw3.player and StreamFrameHub."""

    def __init__(
        self,
        port: int,
        lock: Any,
        frame_width: int,
        frame_height: int,
        fps: int,
        stereo_format: str = "SBS_HALF",
        stream_uri: str = "/stream.jpg",
        stream_content_type: str = "image/jpeg",
        auth: tuple[str, str] | None = None,
        host: str = "",
        args: Any = None,
    ) -> None:
        from ..player.stream_hub import StreamFrameHub

        self.port = port
        self.host = host
        self.lock = lock
        self.op_lock = threading.Lock()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.stereo_format = stereo_format
        self.stream_uri = stream_uri
        self.stream_content_type = stream_content_type
        self.auth = auth
        self.args = args

        self.stream_hub = StreamFrameHub(
            lock=lock,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            stereo_format=stereo_format,
            stream_uri=stream_uri,
            stream_content_type=stream_content_type,
        )
        self.stream_hub.service = self
        self.thread: threading.Thread | None = None
        self.state: Any = None

    def _stop(self):
        self.stream_hub.shutdown_event.set()
        if self.thread is not None:
            if self.thread.ident is not None:
                self.thread.join(timeout=5.0)
            self.thread = None
        if self.state is not None:
            self.state.close()
            self.state = None
        self.stream_hub.shutdown_event.clear()

    def _start(self):
        import asyncio
        import os
        import secrets
        from types import SimpleNamespace

        from hypercorn.config import Config

        from ..player.app import app
        from ..player.dir_config import CERT_DIR
        from ..player.server import CERT_FILE, KEY_FILE, SECRET_KEY_FILE, generate_self_signed_cert, serve_with_watchdog
        from ..player.server_state import ServerState

        bind_host = self.host or "127.0.0.1"
        generate_self_signed_cert(CERT_DIR, CERT_FILE, KEY_FILE, bind_host)
        if not os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "wb") as f:
                f.write(secrets.token_bytes(32))

        user, password = self.auth if self.auth is not None else (None, None)
        self.stream_hub.update_stream_params(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps,
            stereo_format=self.stereo_format,
        )

        args_ns = SimpleNamespace(
            mode="desktop",
            enable_file_library=False,
            stream_hub=self.stream_hub,
            user=user,
            password=password,
            secret_key_file=SECRET_KEY_FILE,
            debug=False,
            debug_console=False,
            state={"stop_event": self.stream_hub.shutdown_event},
        )
        if hasattr(app.state, "server_state") and app.state.server_state is not None:
            app.state.server_state.close()
        self.state = ServerState(args_ns)
        app.state.server_state = self.state
        app.state.media_library = None

        config = Config()
        config.bind = [f"{bind_host}:{self.port}"]
        config.certfile = CERT_FILE
        config.keyfile = KEY_FILE
        config.accesslog = None
        config.errorlog = None
        config.graceful_timeout = 0.5

        def run_server():
            shutdown_event = asyncio.Event()
            asyncio.run(serve_with_watchdog(args_ns, config, shutdown_event))

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop(self):
        with self.op_lock:
            self._stop()

    def start(self):
        with self.op_lock:
            self._stop()
            self._start()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def get_settings(self) -> dict[str, Any]:
        if self.args is None:
            return {}
        from .utils import get_runtime_stream_settings

        return get_runtime_stream_settings(self.args)

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        if self.args is None:
            return {}
        from .utils import update_runtime_stream_settings

        return update_runtime_stream_settings(self.args, updates)

    def set_frame_data(self, frame_data):
        self.stream_hub.set_frame_data(frame_data)

    def get_frame_data(self):
        return self.stream_hub.get_frame_data()

    def get_fps(self):
        return self.stream_hub.get_fps()
