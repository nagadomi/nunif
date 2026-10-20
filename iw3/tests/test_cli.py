import os
from typing import NamedTuple

import pytest

from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data


class CliSetup(NamedTuple):
    image: str
    video: str
    video_hdr: str
    dir: str
    output: str
    cli: list[str]


@pytest.fixture(scope="module")
def cli_setup(tmp_path_factory: pytest.TempPathFactory) -> CliSetup:
    tmp_dir = tmp_path_factory.mktemp("cli")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    return CliSetup(
        image=os.path.join(smoke_dir, "sd.png"),
        video=os.path.join(smoke_dir, "sd.mkv"),
        video_hdr=os.path.join(smoke_dir, "hdr.mkv"),
        dir=smoke_dir,
        output=output_dir,
        cli=["python", "-m", "iw3.cli", "-y", "-o", output_dir],
    )


def test_cli_image(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.image, "--depth-model", "Any_S", "--metadata"])


def test_cli_video(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--metadata"])


def test_cli_dir(cli_setup: CliSetup) -> None:
    run(
        [
            "python",
            "-m",
            "iw3",
            "-y",
            "-i",
            cli_setup.dir,
            "-o",
            cli_setup.output,
            "--depth-model",
            "Any_S",
            "--metadata",
        ]
    )


def test_cli_ema(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
        ]
    )


def test_cli_batch(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
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


def test_cli_low_vram(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--metadata", "--low-vram"])


def test_cli_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
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


def test_cli_vda_stream(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "VDA_Stream_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )


def test_cli_hdr(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
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


def test_cli_hdr_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
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


def test_cli_hdr2sdr(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
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


def test_cli_inpaint(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--method", "forward_inpaint"])


def test_cli_inpaint_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
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


def test_cli_export_import(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "Any_S",
            "--export-disparity",
            "--export-depth-only",
            "--export-depth-fit",
        ]
    )
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
            "--depth-model",
            "VDA_S",
            "--export",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image,
            "--depth-model",
            "Any_S",
            "--export",
            "--depth-aa",
            "--export-depth-fit",
        ]
    )

    hdr_yaml = os.path.join(cli_setup.output, "hdr", "iw3_export.yml")
    sd_yaml = os.path.join(cli_setup.output, "iw3_export.yml")
    if os.path.exists(hdr_yaml):
        run([*cli_setup.cli, "-i", hdr_yaml])
    if os.path.exists(sd_yaml):
        run([*cli_setup.cli, "-i", sd_yaml])


def test_cli_vf(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--vf", "scale=-2:320,crop=256:256"])


def test_cli_keyframe(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--keyframe"])
