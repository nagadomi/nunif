import io
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator

import numpy as np
import pytest
import requests
from PIL import Image

from tests.helpers.runner import run
from waifu2x.download_models import has_models

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_IMAGE_RGB = os.path.join(REPO_ROOT, "tests", "images", "dog_448.png")
TEST_IMAGE_RGBA = os.path.join(REPO_ROOT, "tests", "images", "dog_448_rgba.png")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_WAIFU2X_WEB_TEST") != "1",
        reason="Set RUN_WAIFU2X_WEB_TEST=1 to run waifu2x.web tests",
    ),
]


def find_free_port() -> int:
    """Find an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(base_url: str, proc: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    """Wait until the waifu2x.web server responds on /recaptcha_state.json."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"waifu2x.web exited prematurely (code={proc.returncode}): {stderr_out}")
        try:
            res = requests.get(f"{base_url}/recaptcha_state.json", timeout=1.0)
            if res.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for waifu2x.web at {base_url}")


@pytest.fixture(scope="module")
def web_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[dict[str, str], None, None]:
    """Start a real waifu2x.web Hypercorn server against generated public_html and temp cache."""
    if not has_models():
        run(["python", "-m", "waifu2x.download_models"])

    tmp_dir = tmp_path_factory.mktemp("waifu2x_web")
    public_dir = str(tmp_dir / "public_html")
    cache_dir = str(tmp_dir / "cache")
    cli_out_dir = str(tmp_dir / "cli_out")
    rgb_128_path = str(tmp_dir / "dog_128.png")
    os.makedirs(cli_out_dir, exist_ok=True)

    with Image.open(TEST_IMAGE_RGB) as img:
        img.crop((160, 160, 288, 288)).save(rgb_128_path)

    run(["python", "-m", "waifu2x.web.webgen", "-o", public_dir])

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "waifu2x.web",
            "--bind-addr",
            "127.0.0.1",
            "--port",
            str(port),
            "--root",
            public_dir,
            "--cache-dir",
            cache_dir,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for_server(base_url, proc)
        yield {
            "base_url": base_url,
            "cli_out_dir": cli_out_dir,
            "tmp_dir": str(tmp_dir),
            "rgb_128": rgb_128_path,
            "rgba_448": TEST_IMAGE_RGBA,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize(
    ("case_name", "style", "model_dir", "method", "noise", "scale", "image_format", "input_key"),
    [
        (
            "art_noise1_scale2_png",
            "art",
            "waifu2x/pretrained_models/swin_unet_v3/art",
            "noise_scale",
            1,
            2,
            "png",
            "rgb_128",
        ),
        (
            "art_scan_scale2_png",
            "art_scan",
            "waifu2x/pretrained_models/swin_unet/art_scan",
            "scale",
            -1,
            2,
            "png",
            "rgb_128",
        ),
        (
            "photo_noise0_webp",
            "photo",
            "waifu2x/pretrained_models/swin_unet/photo",
            "noise",
            0,
            -1,
            "webp",
            "rgb_128",
        ),
        (
            "art_rgba_noise2_scale2_png",
            "art",
            "waifu2x/pretrained_models/swin_unet_v3/art",
            "noise_scale",
            2,
            2,
            "png",
            "rgba_448",
        ),
    ],
)
def test_web_matches_cli(
    web_server: dict[str, str],
    case_name: str,
    style: str,
    model_dir: str,
    method: str,
    noise: int,
    scale: int,
    image_format: str,
    input_key: str,
) -> None:
    """Verify waifu2x.web /api output matches waifu2x.cli output."""
    input_path = web_server[input_key]
    cli_output = os.path.join(web_server["cli_out_dir"], f"{case_name}.{image_format}")
    cli_args = [
        "python",
        "-m",
        "waifu2x.cli",
        "--model-dir",
        model_dir,
        "-m",
        method,
        "--format",
        image_format,
        "-i",
        input_path,
        "-o",
        cli_output,
    ]
    if noise >= 0:
        cli_args.extend(["--noise-level", str(noise)])
    run(cli_args)

    format_code = "0" if image_format == "png" else "1"
    with open(input_path, "rb") as f:
        res = requests.post(
            f"{web_server['base_url']}/api",
            files={"file": (os.path.basename(input_path), f, "image/png")},
            data={
                "style": style,
                "noise": str(noise),
                "scale": str(scale),
                "format": format_code,
            },
            timeout=60,
        )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == f"image/{image_format}"

    with Image.open(cli_output) as cli_img, Image.open(io.BytesIO(res.content)) as web_img:
        assert web_img.size == cli_img.size
        assert web_img.mode == cli_img.mode
        cli_arr = np.array(cli_img).astype(int)
        web_arr = np.array(web_img).astype(int)
        abs_diff = np.abs(web_arr - cli_arr)
        max_allowed = 0 if input_key == "rgb_128" else 1
        assert abs_diff.max() <= max_allowed and abs_diff.mean() < 0.01, (
            f"Mismatch for {case_name}: max_abs_diff={abs_diff.max()}, mean={abs_diff.mean():.6f}"
        )
