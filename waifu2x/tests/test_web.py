import io
import os
from collections.abc import Generator
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diskcache import Cache
from fastapi.testclient import TestClient
from PIL import Image

from nunif.utils import pil_io
from tests.helpers.runner import run
from waifu2x.web import server

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_WAIFU2X_WEB_TEST") != "1",
        reason="Set RUN_WAIFU2X_WEB_TEST=1 to run waifu2x.web tests",
    ),
]


class DummyWaifu2x:
    device = torch.device("cpu")

    def convert(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor | None,
        method: str,
        noise_level: int,
        tile_size: int | None = None,
        batch_size: int | None = None,
        tta: bool = False,
        enable_amp: bool = True,
        output_device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if method in {"scale", "noise_scale"}:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                scale_factor=2.0,
                mode="nearest",
            ).squeeze(0)
            if alpha is not None:
                alpha = torch.nn.functional.interpolate(
                    alpha.unsqueeze(0),
                    scale_factor=2.0,
                    mode="nearest",
                ).squeeze(0)
        return x, alpha


@pytest.fixture(scope="module")
def web_env(tmp_path_factory: pytest.TempPathFactory) -> Generator[dict[str, Any], None, None]:
    tmp_dir = tmp_path_factory.mktemp("waifu2x_web_unit")
    public_dir = str(tmp_dir / "public_html")
    cache_dir = str(tmp_dir / "cache")
    run(["python", "-m", "waifu2x.web.webgen", "-o", public_dir])

    cache = Cache(cache_dir, size_limit=64 * 1024 * 1024)
    cache_gc = server.CacheGC(cache, interval=60)
    cfg = ConfigParser()
    cfg["recaptcha"] = {"site_key": "", "secret_key": ""}
    cfg["turnstile"] = {"site_key": "", "secret_key": ""}

    args = SimpleNamespace(
        root=public_dir,
        threads=4,
        url_timeout=2,
        tile_size=64,
        batch_size=1,
        tta=False,
        disable_amp=True,
        image_lib="pil",
        cache_ttl=1,
        enable_recaptcha=False,
        enable_turnstile=False,
        prefer_webp=False,
    )

    dummy_ctx = DummyWaifu2x()

    prev = (
        server.command_args,
        server.config,
        server.art_ctx,
        server.art_scan_ctx,
        server.photo_ctx,
        server.cache,
        server.cache_gc,
        server.IL,
        server.MAX_NOISE_PIXELS,
        server.MAX_SCALE_PIXELS,
        server.MAX_BODY_SIZE,
        server.MAX_REQUEST_BODY_SIZE,
    )

    server.command_args = args
    server.config = cfg
    server.art_ctx = dummy_ctx
    server.art_scan_ctx = dummy_ctx
    server.photo_ctx = dummy_ctx
    server.cache = cache
    server.cache_gc = cache_gc
    server.IL = pil_io
    server.MAX_NOISE_PIXELS = 3000 * 3000
    server.MAX_SCALE_PIXELS = 1500 * 1500
    server.MAX_BODY_SIZE = 5
    server.MAX_REQUEST_BODY_SIZE = 6 * server.SIZE_MB

    sample_rgb = tmp_dir / "sample.png"
    Image.new("RGB", (40, 30), color=(32, 64, 128)).save(sample_rgb, format="PNG")

    try:
        with TestClient(server.app) as client:
            yield {
                "client": client,
                "tmp_dir": tmp_dir,
                "sample_rgb": sample_rgb,
                "cache": cache,
            }
    finally:
        cache.close()
        (
            server.command_args,
            server.config,
            server.art_ctx,
            server.art_scan_ctx,
            server.photo_ctx,
            server.cache,
            server.cache_gc,
            server.IL,
            server.MAX_NOISE_PIXELS,
            server.MAX_SCALE_PIXELS,
            server.MAX_BODY_SIZE,
            server.MAX_REQUEST_BODY_SIZE,
        ) = prev


def test_web_static_routing_and_recaptcha_state(web_env: dict[str, Any]) -> None:
    client: TestClient = web_env["client"]

    r_state = client.get("/recaptcha_state.json")
    assert r_state.status_code == 200
    assert r_state.json() == {
        "recaptcha_enabled": False,
        "recaptcha_site_key": "",
        "turnstile_enabled": False,
        "turnstile_site_key": "",
    }

    r_en = client.get("/")
    assert r_en.status_code == 200
    assert '<html lang="en">' in r_en.text

    r_ja = client.get("/", headers={"Accept-Language": "ja,en;q=0.9"})
    assert r_ja.status_code == 200
    assert '<html lang="ja">' in r_ja.text

    assert client.get("/sub/index.html").status_code == 403
    assert client.get("/missing.html").status_code == 404


def test_web_api_conversion_and_get_redirect(web_env: dict[str, Any]) -> None:
    client: TestClient = web_env["client"]
    sample_rgb: Path = web_env["sample_rgb"]

    # 2x scale PNG
    with open(sample_rgb, "rb") as f:
        r_2x = client.post(
            "/api",
            files={"file": (os.path.basename(sample_rgb), f, "image/png")},
            data={"style": "art", "noise": "0", "scale": "2", "format": "0"},
        )
    assert r_2x.status_code == 200
    assert r_2x.headers["content-type"] == "image/png"
    assert "sample_waifu2x_art_noise0_scale.png" in r_2x.headers.get("content-disposition", "")
    with Image.open(io.BytesIO(r_2x.content)) as out_2x:
        assert out_2x.size == (80, 60)

    # 1.6x scale WebP
    with open(sample_rgb, "rb") as f:
        r_16x = client.post(
            "/api",
            files={"file": (os.path.basename(sample_rgb), f, "image/png")},
            data={"style": "photo", "noise": "-1", "scale": "1", "format": "1"},
        )
    assert r_16x.status_code == 200
    assert r_16x.headers["content-type"] == "image/webp"
    assert "sample_waifu2x_photo_scale.webp" in r_16x.headers.get("content-disposition", "")
    with Image.open(io.BytesIO(r_16x.content)) as out_16x:
        assert out_16x.size == (64, 48)

    # GET /api always redirects to /
    r_get = client.get("/api", follow_redirects=False)
    assert r_get.status_code == 302
    assert r_get.headers.get("location") == "/"


def test_web_api_validation_and_limits(web_env: dict[str, Any], tmp_path: Path) -> None:
    client: TestClient = web_env["client"]
    sample_rgb: Path = web_env["sample_rgb"]

    assert server.is_public_ip("127.0.0.1") is False
    assert server.is_public_ip("::1") is False
    assert server.is_public_ip("192.168.1.1") is False
    assert server.is_public_ip("169.254.169.254") is False
    assert server.is_public_ip("8.8.8.8") is True

    # Invalid parameters -> 400
    with open(sample_rgb, "rb") as f:
        r_bad_param = client.post(
            "/api",
            files={"file": ("sample.png", f, "image/png")},
            data={"style": "invalid_style", "noise": "0", "scale": "2", "format": "0"},
        )
    assert r_bad_param.status_code == 400

    # Missing file and URL -> 400
    r_empty = client.post("/api", data={"style": "art", "noise": "0", "scale": "-1", "format": "0"})
    assert r_empty.status_code == 400

    huge_path = tmp_path / "huge.png"
    Image.new("RGB", (3200, 3200), color="white").save(huge_path, format="PNG")
    with open(huge_path, "rb") as f:
        r_huge = client.post(
            "/api",
            files={"file": ("huge.png", f, "image/png")},
            data={"style": "art", "noise": "-1", "scale": "-1", "format": "0"},
        )
    assert r_huge.status_code == 413

    r_big_body = client.post(
        "/api",
        content=b"x" * (6 * 1024 * 1024 + 1024),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r_big_body.status_code == 413
