import os

import pytest

from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data

MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"


@pytest.fixture(scope="module")
def cli_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_dir = tmp_path_factory.mktemp("cli")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    run(["python", "-m", "waifu2x.download_models"])
    return {
        "image": os.path.join(smoke_dir, "sd.png"),
        "video": os.path.join(smoke_dir, "sd.mkv"),
        "dir": smoke_dir,
        "output": output_dir,
    }


def test_cli_image(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["image"],
            "--noise-level",
            "0",
            "-m",
            "noise_scale",
        ]
    )


def test_cli_video(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["video"],
            "--noise-level",
            "1",
            "-m",
            "noise",
        ]
    )


def test_cli_dir(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["dir"],
            "-m",
            "scale",
        ]
    )
