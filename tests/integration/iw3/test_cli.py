import os

import pytest

from tests.helpers.runner import run

OUTPUT_DIR = "tests/data/smoke/iw3"
CLI = ["python", "-m", "iw3.cli", "-y", "-o", OUTPUT_DIR]


@pytest.fixture(scope="module")
def iw3_cli_setup(smoke_data_dir: str) -> dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return {
        "image": os.path.join(smoke_data_dir, "sd.png"),
        "video": os.path.join(smoke_data_dir, "sd.mkv"),
        "video_hdr": os.path.join(smoke_data_dir, "hdr.mkv"),
        "dir": smoke_data_dir,
        "output": OUTPUT_DIR,
    }


def test_cli_image(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["image"], "--depth-model", "Any_S", "--metadata"])


def test_cli_video(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["video"], "--depth-model", "Any_S", "--metadata"])


def test_cli_dir(iw3_cli_setup: dict[str, str]) -> None:
    out = iw3_cli_setup["output"]
    run(["python", "-m", "iw3", "-y", "-i", iw3_cli_setup["dir"], "-o", out, "--depth-model", "Any_S", "--metadata"])


def test_cli_ema(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "Any_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
        ]
    )


def test_cli_batch(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "Any_S",
            "--metadata",
            "--batch-size",
            "4",
            "--max-workers",
            "2",
            "--cuda-stream",
        ]
    )


def test_cli_low_vram(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["video"], "--depth-model", "Any_S", "--metadata", "--low-vram"])


def test_cli_vda(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "VDA_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--disable-scene-cache",
        ]
    )


def test_cli_vda_stream(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "VDA_Stream_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )


def test_cli_hdr(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video_hdr"],
            "--depth-model",
            "Any_S",
            "--colorspace",
            "auto",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p10le",
        ]
    )


def test_cli_hdr_vda(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video_hdr"],
            "--depth-model",
            "VDA_S",
            "--colorspace",
            "auto",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p10le",
        ]
    )


def test_cli_hdr2sdr(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video_hdr"],
            "--depth-model",
            "Any_S",
            "--colorspace",
            "bt709-tv",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p",
        ]
    )


def test_cli_inpaint(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["video"], "--depth-model", "Any_S", "--method", "forward_inpaint"])


def test_cli_inpaint_vda(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "VDA_S",
            "--method",
            "mlbw_l2_inpaint",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )


def test_cli_export_import(iw3_cli_setup: dict[str, str]) -> None:
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video"],
            "--depth-model",
            "Any_S",
            "--export-disparity",
            "--export-depth-only",
            "--export-depth-fit",
        ]
    )
    run(
        [
            *CLI,
            "-i",
            iw3_cli_setup["video_hdr"],
            "--depth-model",
            "VDA_S",
            "--export",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )
    run([*CLI, "-i", iw3_cli_setup["image"], "--depth-model", "Any_S", "--export", "--depth-aa", "--export-depth-fit"])

    hdr_yaml = os.path.join(iw3_cli_setup["output"], "hdr", "iw3_export.yml")
    sd_yaml = os.path.join(iw3_cli_setup["output"], "iw3_export.yml")
    if os.path.exists(hdr_yaml):
        run([*CLI, "-i", hdr_yaml])
    if os.path.exists(sd_yaml):
        run([*CLI, "-i", sd_yaml])


def test_cli_vf(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["video"], "--depth-model", "Any_S", "--vf", "scale=-2:320,crop=256:256"])


def test_cli_keyframe(iw3_cli_setup: dict[str, str]) -> None:
    run([*CLI, "-i", iw3_cli_setup["video"], "--depth-model", "Any_S", "--keyframe"])
