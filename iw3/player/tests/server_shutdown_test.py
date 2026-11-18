import argparse
import asyncio
import gc
import socket
import ssl
import threading
import time
import warnings

import h2.config
import h2.connection
import h2.events
import pytest

from ..app import app
from ..server import create_hypercorn_config, generate_self_signed_cert, serve_with_watchdog
from ..server_state import ServerState
from ..stream_hub import StreamFrameHub


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _DummyDesktopService:
    def __init__(self, stream_hub: StreamFrameHub) -> None:
        self.stream_hub = stream_hub
        self.stream_hub.service = self
        self.running = True
        self.stop_called = False

    def is_running(self) -> bool:
        return self.running

    def ensure_running(self, timeout: float = 30.0) -> None:
        self.running = True

    def stop(self) -> None:
        self.stop_called = True
        self.running = False
        self.stream_hub.clear_frames(stop_clients=True)


def _setup_test_server(tmp_path, use_tls: bool = True):
    port = _find_free_port()
    hub = StreamFrameHub(frame_width=640, frame_height=360, fps=30, stereo_format="SBS_HALF")
    hub.set_frame_data((b"frame0", time.perf_counter()))
    service = _DummyDesktopService(hub)
    stop_event = threading.Event()

    secret_file = tmp_path / "secret.key"
    secret_file.write_bytes(b"16bytessecretkey")

    args = argparse.Namespace(
        mode="player",
        root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        enable_file_library=False,
        stream_hub=hub,
        desktop_service=service,
        secret_key_file=str(secret_file),
        state={"stop_event": stop_event},
    )
    state = ServerState(args)

    if use_tls:
        cert_file = str(tmp_path / "server.crt")
        key_file = str(tmp_path / "server.key")
        generate_self_signed_cert(str(tmp_path), cert_file, key_file, "127.0.0.1")
    else:
        cert_file = None
        key_file = None

    config = create_hypercorn_config(
        bind_addr="127.0.0.1",
        port=port,
        cert_file=cert_file,
        key_file=key_file,
        graceful_timeout=0.5,
    )

    return args, state, config, hub, service, stop_event, port


async def _connect_h2(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    ssl_ctx.set_alpn_protocols(["h2"])

    for _ in range(50):
        try:
            return await asyncio.open_connection("127.0.0.1", port, ssl=ssl_ctx)
        except OSError:
            await asyncio.sleep(0.05)
    raise AssertionError("Server did not become ready in time")


async def _read_until_marker(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    conn: h2.connection.H2Connection,
    marker: bytes,
) -> tuple[int | None, bytes]:
    status_code: int | None = None
    body = bytearray()
    while marker not in body:
        data = await reader.read(65535)
        if not data:
            break
        for event in conn.receive_data(data):
            if isinstance(event, h2.events.ResponseReceived):
                for name, value in event.headers:
                    if name == b":status":
                        status_code = int(value)
            elif isinstance(event, h2.events.DataReceived):
                body.extend(event.data)
                conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
        pending_send = conn.data_to_send()
        if pending_send:
            writer.write(pending_send)
            await writer.drain()
    return status_code, bytes(body)


@pytest.mark.parametrize("trigger_mode", ["stop_event", "cancel"])
def test_serve_with_watchdog_graceful_shutdown_while_streaming(tmp_path, trigger_mode: str):
    async def _run() -> None:
        args, state, config, hub, service, stop_event, port = _setup_test_server(tmp_path, use_tls=True)
        prev_state = getattr(app.state, "server_state", None)
        app.state.server_state = state
        shutdown_event = asyncio.Event()

        server_task = asyncio.create_task(serve_with_watchdog(args, config, shutdown_event))
        try:
            reader, writer = await _connect_h2(port)
            try:
                conn = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=True))
                conn.initiate_connection()
                stream_id = conn.get_next_available_stream_id()
                conn.send_headers(
                    stream_id,
                    [
                        (":method", "GET"),
                        (":path", "/api/stream"),
                        (":authority", f"127.0.0.1:{port}"),
                        (":scheme", "https"),
                    ],
                    end_stream=True,
                )
                writer.write(conn.data_to_send())
                await writer.drain()

                status_code, first_chunk = await asyncio.wait_for(
                    _read_until_marker(reader, writer, conn, b"frame0"),
                    timeout=3.0,
                )
                assert status_code == 200
                assert b"frame0" in first_chunk

                if trigger_mode == "stop_event":
                    stop_event.set()
                else:
                    server_task.cancel()
                await asyncio.wait_for(server_task, timeout=5.0)
            finally:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except Exception:
                    pass

            assert hub.shutdown_event.is_set()
            assert service.stop_called
        finally:
            if not server_task.done():
                stop_event.set()
                shutdown_event.set()
                try:
                    await asyncio.wait_for(server_task, timeout=5.0)
                except BaseException:
                    pass
            state.close()
            app.state.server_state = prev_state

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        asyncio.run(_run())
        gc.collect()

    unawaited = [
        str(w.message)
        for w in recorded
        if issubclass(w.category, RuntimeWarning) and "was never awaited" in str(w.message)
    ]
    assert unawaited == []
