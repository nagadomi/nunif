import os

import pytest

from tests.helpers.runner import run

OUTPUT_DIR = "tests/data/smoke/waifu2x"
MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"
CLI = ["python", "-m", "waifu2x.cli", "-y", "-o", OUTPUT_DIR, "--model-dir", MODEL_DIR]


@pytest.fixture(scope="module")
def waifu2x_cli_setup(smoke_data_dir: str) -> dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run(["python", "-m", "waifu2x.download_models"])
    return {
        "image": os.path.join(smoke_data_dir, "sd.png"),
        "video": os.path.join(smoke_data_dir, "sd.mkv"),
        "dir": smoke_data_dir,
    }


def test_cli_image(waifu2x_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", waifu2x_cli_setup["image"], "--noise-level", "0", "-m", "noise_scale"])


def test_cli_video(waifu2x_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", waifu2x_cli_setup["video"], "--noise-level", "1", "-m", "noise"])


def test_cli_dir(waifu2x_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", waifu2x_cli_setup["dir"], "-m", "scale"])
