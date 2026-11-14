import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest
from diskcache import Cache
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from ..app import app
from ..media_library import MediaLibrary
from ..server_state import ServerState
from ..stream_hub import StreamFrameHub


@pytest.fixture
def test_setup(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_bytes(b"16bytessecretkey")

    # Create dummy media files
    img = PILImage.new("RGB", (200, 100), color="blue")
    img.save(media_dir / "sample_LRF.png", format="PNG")

    state = SimpleNamespace(
        image_root=str(media_dir),
        auth_user=None,
        auth_password=None,
        debug_mode=False,
        debug_console=False,
        cache_db=Cache(str(cache_dir)),
        secret_key_file=str(secret_file),
    )
    lib = MediaLibrary(state)

    # Attach to app.state
    app.state.server_state = state
    app.state.media_library = lib

    client = TestClient(app)
    yield SimpleNamespace(client=client, state=state, lib=lib, media_dir=media_dir)
    state.cache_db.close()


def test_api_files(test_setup):
    client = test_setup.client
    res = client.get("/api/files?path=/")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "sample_LRF.png"
    assert data[0]["stereo_format"] == "SBS_FULL"
    assert data[0]["type"] == "image"

    # Deprecated /api/list should now return 404
    assert client.get("/api/list?path=/").status_code == 404


def test_api_file_info(test_setup):
    client = test_setup.client
    # Tagged file
    res = client.get("/api/file/info?path=sample_LRF.png")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "sample_LRF.png"
    assert data["stereo_format"] == "SBS_FULL"
    assert data["type"] == "image"

    # Untagged file created on the fly
    plain_img = PILImage.new("RGB", (64, 64), color="red")
    plain_img.save(test_setup.media_dir / "plain.png", format="PNG")

    # Untagged returns None in list_files
    res_list = client.get("/api/files?path=/")
    plain_item = next(it for it in res_list.json() if it["name"] == "plain.png")
    assert plain_item["stereo_format"] is None

    # /api/file/info predicts format (FLAT for uniform color)
    res_info = client.get("/api/file/info?path=plain.png")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert info_data["name"] == "plain.png"
    assert info_data["stereo_format"] == "FLAT"

    # Nonexistent file returns 404
    assert client.get("/api/file/info?path=nonexistent.png").status_code == 404


def test_api_media(test_setup):
    client = test_setup.client
    res = client.get("/api/media?path=sample_LRF.png")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/png")
    assert len(res.content) > 0

    # Nonexistent media file returns 404
    assert client.get("/api/media?path=nonexistent.png").status_code == 404

    # Deprecated /api/image should now return 404
    assert client.get("/api/image?path=sample_LRF.png").status_code == 404


def test_api_thumbnail_and_etag(test_setup):
    client = test_setup.client
    # Initial request should return 200 with an ETag
    res1 = client.get("/api/thumbnail?path=sample_LRF.png")
    assert res1.status_code == 200
    assert res1.headers["content-type"] == "image/webp"
    etag = res1.headers.get("etag")
    assert etag is not None

    # Subsequent request with If-None-Match should return 304 Not Modified
    res2 = client.get("/api/thumbnail?path=sample_LRF.png", headers={"If-None-Match": etag})
    assert res2.status_code == 304


def test_api_key(test_setup):
    client = test_setup.client
    res = client.get("/api/key")
    assert res.status_code == 200
    data = res.json()
    assert "key" in data
    assert isinstance(data["key"], str)


def test_api_cache_clear(test_setup):
    client = test_setup.client
    # Insert something into cache
    test_setup.state.cache_db.set("test_key", b"test_value")
    assert test_setup.state.cache_db.get("test_key") == b"test_value"

    res = client.post("/api/cache/clear")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "message": "Cache cleared"}
    assert test_setup.state.cache_db.get("test_key") is None


def test_api_auth_required(test_setup):
    # Configure authentication on server state
    test_setup.state.auth_user = "admin"
    test_setup.state.auth_password = "password123"

    client = test_setup.client

    # No credentials -> 401
    res = client.get("/api/files?path=/")
    assert res.status_code == 401
    assert "WWW-Authenticate" in res.headers

    # Invalid credentials -> 401
    res_bad = client.get("/api/files?path=/", auth=("admin", "wrong_pass"))
    assert res_bad.status_code == 401

    # Valid credentials -> 200
    res_ok = client.get("/api/files?path=/", auth=("admin", "password123"))
    assert res_ok.status_code == 200


def test_api_files_with_broken_symlinks(test_setup):
    client = test_setup.client
    media_dir = test_setup.media_dir

    # Create a broken link in media_dir (NTFS junction on Windows, symlink on POSIX)
    broken_link = media_dir / "broken_link"
    if sys.platform == "win32":
        import _winapi

        temp_target = media_dir / "temp_target"
        temp_target.mkdir()
        try:
            _winapi.CreateJunction(str(temp_target), str(broken_link))
        except OSError as e:
            temp_target.rmdir()
            pytest.skip(f"Junction creation not permitted in this environment: {e}")
        temp_target.rmdir()
    else:
        try:
            os.symlink(media_dir / "nonexistent", broken_link)
        except OSError as e:
            pytest.skip(f"Symlink creation not permitted in this environment: {e}")

    try:
        # API should return 200 and omit broken link instead of failing with 500
        res = client.get("/api/files?path=/")
        assert res.status_code == 200
        data = res.json()
        names = [item["name"] for item in data]
        assert "sample_LRF.png" in names
        assert "broken_link" not in names
    finally:
        if os.path.lexists(broken_link):
            try:
                os.unlink(broken_link)
            except OSError:
                pass


def test_api_config_player_mode(test_setup):
    client = test_setup.client
    res = client.get("/api/config")
    assert res.status_code == 200
    assert res.json() == {
        "mode": "player",
        "features": {
            "file_library": True,
            "desktop_stream": False,
        },
    }
    assert client.get("/api/stream/info").status_code == 404
    assert client.get("/process_token").status_code == 404
    assert client.get("/stream.jpg").status_code == 404


def test_stream_frame_hub():
    hub = StreamFrameHub(frame_width=1920, frame_height=1080, fps=30, stereo_format="SBS_HALF")
    assert hub.get_frame_data() is None
    assert hub.get_fps() == 0.0

    info = hub.get_stream_info()
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["fps"] == 30
    assert info["stereo_format"] == "SBS_HALF"
    assert info["stream_uri"] == "/stream.jpg"
    assert len(info["process_token"]) == 16

    hub.set_frame_data((b"frame1", 1.0))
    assert hub.get_frame_data() == (b"frame1", 1.0)
    # Consuming clears frame_data_raw while retaining last frame_data
    assert hub.get_frame_data() == (b"frame1", 1.0)

    called = 0

    def lazy_provider():
        nonlocal called
        called += 1
        return (b"frame2", 2.0)

    hub.set_frame_data(lazy_provider)
    assert hub.get_frame_data() == (b"frame2", 2.0)
    assert called == 1
    assert hub.get_frame_data() == (b"frame2", 2.0)
    assert called == 1

    async def read_one_part():
        hub.set_frame_data((b"\xff\xd8\xff\xd9", time.perf_counter() + 10.0))
        gen = hub.async_stream_generator()
        chunk = await anext(gen)
        hub.shutdown_event.set()
        with pytest.raises(StopAsyncIteration):
            await anext(gen)
        return chunk

    part = asyncio.run(read_one_part())
    assert b"--frame\r\n" in part
    assert b"Content-Type: image/jpeg\r\n" in part
    assert b"Content-Length: 4\r\n\r\n\xff\xd8\xff\xd9\r\n" in part


def test_api_desktop_stream_mode(tmp_path):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_bytes(b"16bytessecretkey")
    hub = StreamFrameHub(frame_width=3840, frame_height=1080, fps=60, stereo_format="SBS_FULL")
    state = ServerState(
        SimpleNamespace(
            mode="desktop",
            enable_file_library=False,
            stream_hub=hub,
            secret_key_file=str(secret_file),
        )
    )
    assert state.cache_db is None

    prev_state = getattr(app.state, "server_state", None)
    prev_lib = getattr(app.state, "media_library", None)
    app.state.server_state = state
    app.state.media_library = None
    try:
        client = TestClient(app)

        res_cfg = client.get("/api/config")
        assert res_cfg.status_code == 200
        assert res_cfg.json() == {
            "mode": "desktop",
            "features": {
                "file_library": False,
                "desktop_stream": True,
            },
        }

        res_info = client.get("/api/stream/info")
        assert res_info.status_code == 200
        info = res_info.json()
        assert info["width"] == 3840
        assert info["height"] == 1080
        assert info["fps"] == 60
        assert info["stereo_format"] == "SBS_FULL"
        assert info["process_token"] == hub.process_token

        res_token = client.get("/process_token")
        assert res_token.status_code == 200
        assert res_token.json() == {"token": hub.process_token}

        assert client.get("/api/files?path=/").status_code == 404

        def one_shot_frame():
            hub.shutdown_event.set()
            return (b"\xff\xd8TEST\xff\xd9", 100.0)

        hub.set_frame_data(one_shot_frame)
        res_stream = client.get("/stream.jpg")
        assert res_stream.status_code == 200
        assert "multipart/x-mixed-replace" in res_stream.headers["content-type"]
        assert b"Content-Length: 8\r\n\r\n\xff\xd8TEST\xff\xd9\r\n" in res_stream.content
    finally:
        hub.shutdown_event.set()
        state.close()
        app.state.server_state = prev_state
        app.state.media_library = prev_lib


def test_resolve_player_stereo_format():
    from iw3.desktop.utils import resolve_player_stereo_format

    assert resolve_player_stereo_format(SimpleNamespace(half_sbs=True)) == "SBS_HALF"
    assert resolve_player_stereo_format(SimpleNamespace(full_sbs=True)) == "SBS_FULL"
    assert resolve_player_stereo_format(SimpleNamespace(half_tb=True)) == "TB_HALF"
    assert resolve_player_stereo_format(SimpleNamespace(tb=True)) == "TB_FULL"
    assert resolve_player_stereo_format(SimpleNamespace(cross_eyed=True)) == "SBS_FULL_CROSS"
    assert resolve_player_stereo_format(SimpleNamespace(vr180=True)) == "SBS_180"
    assert resolve_player_stereo_format(SimpleNamespace(anaglyph="dubois")) == "FLAT"
    assert resolve_player_stereo_format(SimpleNamespace()) == "SBS_HALF"


def test_player_create_parser():
    from iw3.player.server import create_parser, set_state_args

    parser = create_parser()
    args = parser.parse_args([])
    assert args.enable_stream is False
    assert args.port == 1304
    set_state_args(args)
    assert "depth_model" not in args.state

    custom = parser.parse_args(
        ["--enable-stream", "--full-sbs", "--stream-fps", "60", "--screenshot", "pil", "--divergence", "2.5"]
    )
    assert custom.enable_stream is True
    assert custom.full_sbs is True
    assert custom.stream_fps == 60
    assert custom.screenshot == "pil"
    assert custom.divergence == 2.5
    assert custom.depth_model == "Any_V2_S"
    assert custom.convergence == 1.0
    assert custom.ema_normalize is True


def test_stream_hub_clear_frames_stops_clients():
    hub = StreamFrameHub(frame_width=1920, frame_height=1080, fps=30)

    async def exercise():
        hub.set_frame_data((b"\xff\xd8A\xff\xd9", time.perf_counter() + 10.0))
        gen = hub.async_stream_generator()
        part = await anext(gen)
        assert b"A" in part
        assert hub.active_clients == 1

        hub.clear_frames(stop_clients=True)
        with pytest.raises(StopAsyncIteration):
            await anext(gen)
        assert hub.active_clients == 0
        assert not hub.shutdown_event.is_set()

    asyncio.run(exercise())


def test_api_player_desktop_stream_lifecycle(tmp_path):
    from iw3.desktop.utils import (
        ensure_default_stereo_format,
        get_runtime_stream_settings,
        read_runtime_adjustment_values,
        resolve_player_stereo_format,
        update_runtime_stream_settings,
    )
    from iw3.player.server import create_parser, set_state_args

    class DummyDesktopService:
        def __init__(self, args, stream_hub):
            self.args = args
            self.stream_hub = stream_hub
            self.stream_hub.service = self
            self.running = False
            self.start_count = 0
            self.stop_count = 0
            ensure_default_stereo_format(self.args)

        def is_running(self):
            return self.running

        def ensure_running(self, timeout=30.0):
            self.start_count += 1
            self.running = True
            self.stream_hub.update_stream_params(
                frame_width=2560,
                frame_height=720,
                fps=int(self.args.stream_fps),
                stereo_format=resolve_player_stereo_format(self.args),
            )

        def stop(self):
            self.stop_count += 1
            self.running = False
            self.stream_hub.clear_frames(stop_clients=True)

        def get_settings(self):
            return get_runtime_stream_settings(self.args)

        def update_settings(self, updates):
            return update_runtime_stream_settings(self.args, updates)

    parser = create_parser()
    cli_args = parser.parse_args(["--enable-stream"])
    set_state_args(cli_args)
    dummy_convergence_model = SimpleNamespace(convergence=cli_args.convergence)
    cli_args.state["convergence_model"] = dummy_convergence_model

    secret_file = tmp_path / "secret.txt"
    secret_file.write_bytes(b"16bytessecretkey")
    hub = StreamFrameHub(frame_width=1920, frame_height=1080, fps=30, stereo_format="SBS_HALF")
    service = DummyDesktopService(cli_args, hub)
    state = ServerState(
        SimpleNamespace(
            mode="player",
            root=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            enable_file_library=True,
            stream_hub=hub,
            desktop_service=service,
            secret_key_file=str(secret_file),
        )
    )
    library = MediaLibrary(state)

    prev_state = getattr(app.state, "server_state", None)
    prev_lib = getattr(app.state, "media_library", None)
    app.state.server_state = state
    app.state.media_library = library
    try:
        client = TestClient(app)

        res_cfg = client.get("/api/config")
        assert res_cfg.status_code == 200
        assert res_cfg.json() == {
            "mode": "player",
            "features": {
                "file_library": True,
                "desktop_stream": True,
            },
        }

        res_info = client.get("/api/stream/info")
        assert res_info.status_code == 200
        info = res_info.json()
        assert info["active"] is True
        assert info["width"] == 2560
        assert info["height"] == 720
        assert service.start_count == 1

        res_stop = client.post("/api/stream/stop")
        assert res_stop.status_code == 200
        assert res_stop.json() == {"active": False}
        assert service.stop_count == 1

        res_start = client.post("/api/stream/start")
        assert res_start.status_code == 200
        assert res_start.json()["active"] is True
        assert service.start_count == 2

        res_get_settings = client.get("/api/stream/settings")
        assert res_get_settings.status_code == 200
        settings = res_get_settings.json()
        assert settings["divergence"] == 1.0
        assert settings["stereo_format"] == "SBS_HALF"

        res_post_settings = client.post(
            "/api/stream/settings",
            json={
                "divergence": 2.5,
                "convergence": 0.75,
                "foreground_scale": -1.5,
                "edge_dilation": 4,
                "stream_fps": 45,
                "stream_quality": 85,
                "synthetic_view": "left",
                "ema_normalize": False,
                "ema_decay": 0.85,
            },
        )
        assert res_post_settings.status_code == 200
        updated = res_post_settings.json()
        assert updated["divergence"] == 2.5
        assert updated["convergence"] == 0.75
        assert dummy_convergence_model.convergence == 0.75
        assert updated["foreground_scale"] == -1.5
        assert updated["edge_dilation"] == 4
        assert read_runtime_adjustment_values(cli_args) == (2.5, 0.75, -1.5, 4, "")
        assert updated["stream_fps"] == 45
        assert updated["stream_quality"] == 85
        assert updated["synthetic_view"] == "left"
        assert updated["ema_normalize"] is False
        assert updated["ema_decay"] == 0.85
    finally:
        state.close()
        app.state.server_state = prev_state
        app.state.media_library = prev_lib

    assert service.is_running() is False
    assert service.stop_count == 2
