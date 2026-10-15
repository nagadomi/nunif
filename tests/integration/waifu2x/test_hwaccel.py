import os

import pytest

from tests.helpers.runner import run

OUTPUT_DIR = "tests/data/smoke_hwaccel/waifu2x"
MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"
HWACCEL = ["--hwaccel", "cuda"]
H264_ENC = "h264_nvenc"
CLI = [
    "python",
    "-m",
    "waifu2x.cli",
    "-y",
    "-o",
    OUTPUT_DIR,
    "--model-dir",
    MODEL_DIR,
    "--video-codec",
    H264_ENC,
    *HWACCEL,
]


@pytest.fixture(scope="module")
def waifu2x_hwaccel_setup(smoke_data_dir: str) -> dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run(["python", "-m", "waifu2x.download_models"])
    return {
        "video": os.path.join(smoke_data_dir, "sd.mkv"),
        "dir": smoke_data_dir,
    }


@pytest.mark.hwaccel
def test_hwaccel_video(waifu2x_hwaccel_setup: dict[str, str]) -> None:
    run([*CLI, "-i", waifu2x_hwaccel_setup["video"], "--noise-level", "1", "-m", "noise"])


@pytest.mark.hwaccel
def test_hwaccel_dir(waifu2x_hwaccel_setup: dict[str, str]) -> None:
    run([*CLI, "-i", waifu2x_hwaccel_setup["dir"], "-m", "scale"])
