import http.server
import os
import threading
from pathlib import Path
from typing import Any

import wx

from nunif.gui.common import TQDMGUI, FPSTracker
from nunif.utils.downloader import CheckpointDownloader


class _DummyResponseHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b"x" * (64 * 1024)
    advertised_length: int | None = None

    def do_GET(self) -> None:
        self.send_response(200)
        length = self.advertised_length if self.advertised_length is not None else len(self.payload)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _start_server(
    payload: bytes, advertised_length: int | None = None
) -> tuple[http.server.ThreadingHTTPServer, threading.Thread, str]:
    class Handler(_DummyResponseHandler):
        pass

    Handler.payload = payload
    Handler.advertised_length = advertised_length
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, thread, f"http://127.0.0.1:{port}/model.safetensors"


def test_checkpoint_downloader_success(tmp_path: Path) -> None:
    payload = os.urandom(128 * 1024 + 123)
    server, thread, url = _start_server(payload)
    try:
        dst_path = tmp_path / "checkpoints" / "model.safetensors"
        events: list[tuple[str, int]] = []

        class FakeTQDM:
            def __init__(self, **kwargs: Any) -> None:
                events.append(("init", kwargs["total"]))

            def update(self, n: int = 1) -> None:
                events.append(("update", n))

            def close(self) -> None:
                events.append(("close", 0))

        dl = CheckpointDownloader(url=url, dst_path=str(dst_path), name="test-model")
        ok = dl.run(block_size=32 * 1024, tqdm_fn=FakeTQDM)

        assert ok is True
        assert dst_path.read_bytes() == payload
        assert [p.name for p in dst_path.parent.iterdir()] == ["model.safetensors"]
        assert events[0] == ("init", len(payload))
        assert sum(v for k, v in events if k == "update") == len(payload)
        assert events[-1] == ("close", 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_checkpoint_downloader_cancel(tmp_path: Path) -> None:
    payload = b"a" * (256 * 1024)
    server, thread, url = _start_server(payload)
    stop_event = threading.Event()
    try:
        dst_path = tmp_path / "checkpoints" / "model.safetensors"

        class CancelOnFirstUpdateTQDM:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def update(self, n: int = 1) -> None:
                stop_event.set()

            def close(self) -> None:
                pass

        dl = CheckpointDownloader(url=url, dst_path=str(dst_path), name="test-model")
        ok = dl.run(block_size=16 * 1024, stop_event=stop_event, tqdm_fn=CancelOnFirstUpdateTQDM)

        assert ok is False
        assert not dst_path.exists()
        assert list(dst_path.parent.iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_checkpoint_downloader_incomplete(tmp_path: Path) -> None:
    import pytest

    payload = b"a" * (32 * 1024)
    server, thread, url = _start_server(payload, advertised_length=64 * 1024)
    try:
        dst_path = tmp_path / "checkpoints" / "model.safetensors"
        dl = CheckpointDownloader(url=url, dst_path=str(dst_path), name="test-model")
        with pytest.raises(RuntimeError, match="Incomplete download"):
            dl.run(block_size=16 * 1024, show_progress=False)

        assert not dst_path.exists()
        assert list(dst_path.parent.iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_tqdm_gui_byte_scaling(monkeypatch: Any) -> None:
    posted: list[Any] = []
    monkeypatch.setattr(wx, "PostEvent", lambda parent, event: posted.append(event))

    bar = TQDMGUI(None, total=3 * 1024 * 1024 * 1024 + 500, desc="Downloading", unit="iB")
    bar.update(512)
    bar.update(512)
    bar.update(2048)
    bar.close()

    assert posted[0].GetValue() == (0, 3 * 1024 * 1024, "Downloading")
    assert posted[0].unit == "KiB"
    assert posted[1].GetValue() == (1, 1, "Downloading")
    assert posted[2].GetValue() == (1, 2, "Downloading")
    assert posted[3].GetValue() == (2, 0, "Downloading")


def test_fps_tracker_kib_formatting() -> None:
    tracker = FPSTracker(update_interval=0.0)
    tracker.reset(start_pos=0, unit="KiB")
    assert tracker.update(0, 2048, "Downloading") is None
    tracker.history[0] = (tracker.history[0][0] - 2.0, 0)
    status = tracker.update(1024, 2048, "Downloading")
    assert status is not None
    assert "MiB" in status
    assert "MiB/s" in status

    tracker.reset(start_pos=1024)
    assert tracker.unit == "KiB"


def test_depth_model_auto_download_urls() -> None:
    from iw3.base_depth_model import BaseDepthModel

    classes = BaseDepthModel.get_registered_classes()
    assert len(classes) > 0
    for subcls in classes:
        for model_type in subcls.get_model_types():
            url = subcls.get_model_url(model_type)
            if subcls.has_auto_download(model_type):
                assert url is not None and url.startswith("https://"), f"Missing URL for {model_type}"
            else:
                assert url is None, f"Unexpected URL for manual-only model {model_type}"
