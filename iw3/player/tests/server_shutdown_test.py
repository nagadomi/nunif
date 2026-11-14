import argparse
import asyncio
import gc
import socket
import threading
import time
import warnings

import httpx
import pytest
from hypercorn.config import Config

from ..app import app
from ..server import generate_self_signed_cert, serve_with_watchdog
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

    config = Config()
    config.bind = [f"127.0.0.1:{port}"]
    config.accesslog = None
    config.errorlog = None
    config.graceful_timeout = 0.5
    if use_tls:
        cert_file = str(tmp_path / "server.crt")
        key_file = str(tmp_path / "server.key")
        generate_self_signed_cert(str(tmp_path), cert_file, key_file, "127.0.0.1")
        config.certfile = cert_file
        config.keyfile = key_file

    scheme = "https" if use_tls else "http"
    base_url = f"{scheme}://127.0.0.1:{port}"
    return args, state, config, hub, service, stop_event, base_url


async def _wait_until_ready(base_url: str) -> None:
    async with httpx.AsyncClient(verify=False, http2=True) as client:
        for _ in range(50):
            try:
                res = await client.get(f"{base_url}/api/config")
                if res.status_code == 200:
                    return
            except httpx.HTTPError:
                await asyncio.sleep(0.05)
    raise AssertionError("Server did not become ready in time")


@pytest.mark.parametrize("trigger_mode", ["stop_event", "cancel"])
def test_serve_with_watchdog_graceful_shutdown_while_streaming(tmp_path, trigger_mode: str):
    async def _run() -> None:
        args, state, config, hub, service, stop_event, base_url = _setup_test_server(tmp_path, use_tls=True)
        prev_state = getattr(app.state, "server_state", None)
        app.state.server_state = state
        shutdown_event = asyncio.Event()

        server_task = asyncio.create_task(serve_with_watchdog(args, config, shutdown_event))
        try:
            await _wait_until_ready(base_url)

            async with httpx.AsyncClient(verify=False, http2=True) as client:
                async with client.stream("GET", f"{base_url}/api/stream") as response:
                    assert response.status_code == 200
                    chunk_iter = response.aiter_bytes()
                    first_chunk = await asyncio.wait_for(anext(chunk_iter), timeout=3.0)
                    assert b"frame0" in first_chunk

                    if trigger_mode == "stop_event":
                        stop_event.set()
                    else:
                        server_task.cancel()
                    await asyncio.wait_for(server_task, timeout=5.0)

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
