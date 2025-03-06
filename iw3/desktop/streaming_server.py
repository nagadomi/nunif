""" Mixed-Replace(MJPEG) Streaming Server
"""
import sys
import time
import threading
from string import Template
import io
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
import random
import json
import base64
from collections import deque


STATUS_OK = "200 OK"


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    pass


class StreamingServer():
    def __init__(
            self, port, lock,
            frame_width, frame_height, fps,
            index_template,
            stream_uri="/stream.jpg", stream_content_type="image/jpeg",
            auth=None, host=""):
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

        self.server = None
        self.thread = None
        self.process_token = None
        self.shutdown_event = threading.Event()
        self.fps_counter = deque(maxlen=300)

        if auth is not None:
            user, password = auth
            self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        else:
            self.auth = None

    def _stop(self):
        self.shutdown_event.set()
        if self.server is not None:
            self.server.shutdown()
            self.thread.join()
            self.server.shutdown()
            self.thread = None
            self.server = None
        self.shutdown_event.clear()
        time.sleep(0.1)

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
        with self.lock:
            self.frame_data = frame_data

    def get_frame_data(self):
        with self.lock:
            frame_data = self.frame_data
            self.frame_data = None  # handled
            if callable(frame_data):
                frame_data = frame_data()
            assert isinstance(frame_data, (type(None), bytes))
            return frame_data

    def get_fps(self):
        diff = []
        with self.op_lock:
            prev = None
            for t in self.fps_counter:
                if prev is not None:
                    diff.append(t - prev)
                prev = t
        if diff:
            return round(1.0 / (sum(diff) / len(diff)), 2)
        else:
            return None

    def send_image_stream(self, start_response):
        def gen():
            frame = None
            send_at = time.time()
            bio = io.BytesIO()
            bio.write(b'--frame\r\n' + f"Content-Type: {self.stream_content_type}".encode() + b'\r\n\r\n')
            pos = bio.tell()
            while True:
                try:
                    frame = self.get_frame_data()
                    if frame:
                        with self.op_lock:
                            self.fps_counter.append(time.time())
                        bio.seek(pos, io.SEEK_SET)
                        bio.truncate(pos)
                        bio.write(frame)
                        yield bio.getbuffer().tobytes()
                    if self.shutdown_event.is_set():
                        break
                    if False:  # True if needed
                        now = time.time()
                        if now - send_at < self.delay:
                            time.sleep(self.delay - (now - send_at))
                        send_at = now
                    else:
                        # busy waiting
                        time.sleep(1 / 1000)
                except:  # noqa
                    print("StreamingServer", sys.exc_info(), file=sys.stderr)
                    break
            yield b""

        start_response(
            STATUS_OK,
            [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
        return gen()

    def send_index(self, start_response):
        template = Template(self.index_template)
        page_data = template.substitute(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps,
            stream_uri=self.stream_uri
        ).encode()
        start_response(STATUS_OK, [('Content-type', "text/html; charset=utf-8")])
        return [page_data]

    def send_404(self, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"404 Not Found"]

    def send_process_token(self, start_response):
        start_response(STATUS_OK, [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"token": self.process_token}).encode()]

    def handle(self, environ, start_response):
        uri = environ['PATH_INFO']
        #  print("request", uri)

        if self.auth is not None:
            # HTTP Basic Authentication
            auth = environ.get("HTTP_AUTHORIZATION", "")
            #  print("auth", auth)
            if auth != self.auth:
                start_response(
                    "401 Unauthorized",
                    [("WWW-Authenticate", "Basic charset=utf-8"),
                     ("Content-Type", "text/plain; charset=utf-8")])
                return [b"Authorization Required"]

        if uri == "/":
            return self.send_index(start_response)
        elif uri == "/process_token":
            return self.send_process_token(start_response)
        elif uri == self.stream_uri:
            return self.send_image_stream(start_response)
        else:
            return self.send_404(start_response)
