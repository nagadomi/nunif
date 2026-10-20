import os
from types import SimpleNamespace

import pytest
from diskcache import Cache
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from ..app import app
from ..media_library import MediaLibrary


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

    # Create a broken symlink in media_dir
    broken_link = media_dir / "broken.jpg"
    os.symlink(media_dir / "nonexistent.jpg", broken_link)

    # API should return 200 and omit broken symlink instead of failing with 500
    res = client.get("/api/files?path=/")
    assert res.status_code == 200
    data = res.json()
    names = [item["name"] for item in data]
    assert "sample_LRF.png" in names
    assert "broken.jpg" not in names
