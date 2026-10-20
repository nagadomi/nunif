import os

import pytest

from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data

MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"
HWACCEL = ["--hwaccel", "cuda"]
H264_ENC = "h264_nvenc"


@pytest.fixture(scope="module")
def hwaccel_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_dir = tmp_path_factory.mktemp("hwaccel")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    run(["python", "-m", "waifu2x.download_models"])
    return {
        "video": os.path.join(smoke_dir, "sd.mkv"),
        "dir": smoke_dir,
        "output": output_dir,
    }


@pytest.mark.hwaccel
def test_hwaccel_video(hwaccel_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            hwaccel_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "--video-codec",
            H264_ENC,
            *HWACCEL,
            "-i",
            hwaccel_setup["video"],
            "--noise-level",
            "1",
            "-m",
            "noise",
        ]
    )


@pytest.mark.hwaccel
def test_hwaccel_dir(hwaccel_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            hwaccel_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "--video-codec",
            H264_ENC,
            *HWACCEL,
            "-i",
            hwaccel_setup["dir"],
            "-m",
            "scale",
        ]
    )
