import os
from typing import NamedTuple

import pytest

from tests.helpers.hwaccel import (
    cuda_hwaccel_required,
    nvenc_h264_required,
    nvenc_h265_required,
    vaapi_h264_required,
    vaapi_h265_required,
)
from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data

H264_ENC = "h264_nvenc"
H265_ENC = "hevc_nvenc"
HWACCEL = ["--hwaccel", "cuda"]
H264_OPT = ["--video-codec", H264_ENC, *HWACCEL]
H265_OPT = ["--video-codec", H265_ENC, *HWACCEL]

VAAPI_H264_ENC = "h264_vaapi"
VAAPI_H265_ENC = "hevc_vaapi"
VAAPI_HWACCEL = ["--hwaccel", "vaapi"]
VAAPI_H264_OPT = ["--video-codec", VAAPI_H264_ENC, *VAAPI_HWACCEL]
VAAPI_H265_OPT = ["--video-codec", VAAPI_H265_ENC, *VAAPI_HWACCEL]


class HwaccelSetup(NamedTuple):
    video: str
    video_hdr: str
    dir: str
    output: str
    yaml: str
    cli: list[str]


@pytest.fixture(scope="module")
def hwaccel_setup(tmp_path_factory: pytest.TempPathFactory) -> HwaccelSetup:
    tmp_dir = tmp_path_factory.mktemp("hwaccel")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    return HwaccelSetup(
        video=os.path.join(smoke_dir, "sd.mkv"),
        video_hdr=os.path.join(smoke_dir, "hdr.mkv"),
        dir=smoke_dir,
        output=output_dir,
        yaml=os.path.join(output_dir, "sd", "iw3_export.yml"),
        cli=["python", "-m", "iw3.cli", "-y", "-o", output_dir],
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_video(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--metadata", *H264_OPT])


@pytest.mark.hwaccel
@nvenc_h265_required
def test_hwaccel_dir(hwaccel_setup: HwaccelSetup) -> None:
    test_dir = hwaccel_setup.dir
    out = hwaccel_setup.output
    run(["python", "-m", "iw3", "-y", "-i", test_dir, "-o", out, "--depth-model", "Any_S", "--metadata", *H265_OPT])


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_ema(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_batch(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--batch-size",
            "4",
            "--max-workers",
            "2",
            "--cuda-stream",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_low_vram(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--metadata", "--low-vram", *H264_OPT])


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_vda(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "VDA_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--disable-scene-cache",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_vda_stream(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "VDA_Stream_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h265_required
def test_hwaccel_hdr(hwaccel_setup: HwaccelSetup) -> None:
    video_hdr = hwaccel_setup.video_hdr
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video_hdr,
            "--depth-model",
            "Any_S",
            "--colorspace",
            "auto",
            "--pix-fmt",
            "yuv420p10le",
            *H265_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h265_required
def test_hwaccel_hdr_vda(hwaccel_setup: HwaccelSetup) -> None:
    video_hdr = hwaccel_setup.video_hdr
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video_hdr,
            "--depth-model",
            "VDA_S",
            "--colorspace",
            "auto",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--pix-fmt",
            "yuv420p10le",
            *H265_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_hdr2sdr(hwaccel_setup: HwaccelSetup) -> None:
    video_hdr = hwaccel_setup.video_hdr
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video_hdr,
            "--depth-model",
            "Any_S",
            "--colorspace",
            "bt709-tv",
            "--pix-fmt",
            "yuv420p",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_inpaint(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--method", "forward_inpaint", *H264_OPT])


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_inpaint_vda(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "VDA_S",
            "--method",
            "mlbw_l2_inpaint",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            *H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@nvenc_h264_required
def test_hwaccel_export_import(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    video_hdr = hwaccel_setup.video_hdr
    test_yaml = hwaccel_setup.yaml

    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--export-disparity", *HWACCEL])
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video_hdr,
            "--depth-model",
            "VDA_S",
            "--export",
            "--export-depth-only",
            "--export-depth-fit",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            *HWACCEL,
        ]
    )
    if os.path.exists(test_yaml):
        run([*hwaccel_setup.cli, "-i", test_yaml, "--video-codec", H264_ENC])


@pytest.mark.hwaccel
@cuda_hwaccel_required
def test_hwaccel_vf(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--vf", "scale=-2:320,crop=256:256", *HWACCEL])


@pytest.mark.hwaccel
@cuda_hwaccel_required
def test_hwaccel_keyframe(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--keyframe", *HWACCEL])


@pytest.mark.hwaccel
@vaapi_h264_required
def test_hwaccel_vaapi_video(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--metadata", *VAAPI_H264_OPT])


@pytest.mark.hwaccel
@vaapi_h264_required
def test_hwaccel_vaapi_batch(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--batch-size",
            "4",
            "--max-workers",
            "2",
            *VAAPI_H264_OPT,
        ]
    )


@pytest.mark.hwaccel
@vaapi_h265_required
def test_hwaccel_vaapi_hdr(hwaccel_setup: HwaccelSetup) -> None:
    video_hdr = hwaccel_setup.video_hdr
    run(
        [
            *hwaccel_setup.cli,
            "-i",
            video_hdr,
            "--depth-model",
            "Any_S",
            "--colorspace",
            "auto",
            "--pix-fmt",
            "yuv420p10le",
            *VAAPI_H265_OPT,
        ]
    )


@pytest.mark.hwaccel
@vaapi_h264_required
def test_hwaccel_vaapi_inpaint(hwaccel_setup: HwaccelSetup) -> None:
    video = hwaccel_setup.video
    run([*hwaccel_setup.cli, "-i", video, "--depth-model", "Any_S", "--method", "forward_inpaint", *VAAPI_H264_OPT])
