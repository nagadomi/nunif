import os

import pytest

from tests.helpers.runner import run

OUTPUT_DIR = "tests/data/smoke_hwaccel/iw3"
H264_ENC = "h264_nvenc"
H265_ENC = "hevc_nvenc"
HWACCEL = "--hwaccel cuda"


@pytest.fixture(scope="module")
def iw3_hwaccel_setup(smoke_data_dir: str) -> dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return {
        "video": os.path.join(smoke_data_dir, "sd.mkv"),
        "video_hdr": os.path.join(smoke_data_dir, "hdr.mkv"),
        "dir": smoke_data_dir,
        "output": OUTPUT_DIR,
        "yaml": os.path.join(OUTPUT_DIR, "sd", "iw3_export.yml"),
        "h264_opt": f"--video-codec {H264_ENC} {HWACCEL}",
        "h265_opt": f"--video-codec {H265_ENC} {HWACCEL}",
        "cli": f"python -m iw3.cli -y -o {OUTPUT_DIR}",
    }


@pytest.mark.hwaccel
def test_hwaccel_video(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model Any_S --metadata {h264}")


@pytest.mark.hwaccel
def test_hwaccel_dir(iw3_hwaccel_setup: dict[str, str]) -> None:
    test_dir = iw3_hwaccel_setup["dir"]
    out = iw3_hwaccel_setup["output"]
    h265 = iw3_hwaccel_setup["h265_opt"]
    run(f"python -m iw3 -y -i {test_dir} -o {out} --depth-model Any_S --metadata {h265}")


@pytest.mark.hwaccel
def test_hwaccel_ema(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10 {h264}")


@pytest.mark.hwaccel
def test_hwaccel_batch(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream {h264}")


@pytest.mark.hwaccel
def test_hwaccel_low_vram(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model Any_S --metadata --low-vram {h264}")


@pytest.mark.hwaccel
def test_hwaccel_vda(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(
        f"{cli} -i {video} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10"
        f" --scene-detect --disable-scene-cache {h264}"
    )


@pytest.mark.hwaccel
def test_hwaccel_vda_stream(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect {h264}")


@pytest.mark.hwaccel
def test_hwaccel_hdr(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video_hdr = iw3_hwaccel_setup["video_hdr"]
    h265 = iw3_hwaccel_setup["h265_opt"]
    run(f"{cli} -i {video_hdr} --depth-model Any_S --colorspace auto --pix-fmt yuv420p10le {h265}")


@pytest.mark.hwaccel
def test_hwaccel_hdr_vda(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video_hdr = iw3_hwaccel_setup["video_hdr"]
    h265 = iw3_hwaccel_setup["h265_opt"]
    run(
        f"{cli} -i {video_hdr} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10"
        f" --scene-detect --pix-fmt yuv420p10le {h265}"
    )


@pytest.mark.hwaccel
def test_hwaccel_hdr2sdr(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video_hdr = iw3_hwaccel_setup["video_hdr"]
    h265 = iw3_hwaccel_setup["h265_opt"]
    run(f"{cli} -i {video_hdr} --depth-model Any_S --colorspace bt709-tv --pix-fmt yuv420p {h265}")


@pytest.mark.hwaccel
def test_hwaccel_inpaint(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h264 = iw3_hwaccel_setup["h264_opt"]
    run(f"{cli} -i {video} --depth-model Any_S --method forward_inpaint {h264}")


@pytest.mark.hwaccel
def test_hwaccel_inpaint_vda(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    h265 = iw3_hwaccel_setup["h265_opt"]
    run(
        f"{cli} -i {video} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10"
        f" --scene-detect {h265}"
    )


@pytest.mark.hwaccel
def test_hwaccel_export_import(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    video_hdr = iw3_hwaccel_setup["video_hdr"]
    test_yaml = iw3_hwaccel_setup["yaml"]

    run(f"{cli} -i {video} --depth-model Any_S --export-disparity {HWACCEL}")
    run(
        f"{cli} -i {video_hdr} --depth-model VDA_S --export --export-depth-only --export-depth-fit"
        f" --ema-normalize --ema-buffer 10 --scene-detect {HWACCEL}"
    )
    if os.path.exists(test_yaml):
        run(f"{cli} -i {test_yaml} --video-codec {H264_ENC}")


@pytest.mark.hwaccel
def test_hwaccel_vf(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    run(f'{cli} -i {video} --depth-model Any_S --vf "scale=-2:320,crop=256:256" {HWACCEL}')


@pytest.mark.hwaccel
def test_hwaccel_keyframe(iw3_hwaccel_setup: dict[str, str]) -> None:
    cli = iw3_hwaccel_setup["cli"]
    video = iw3_hwaccel_setup["video"]
    run(f"{cli} -i {video} --depth-model Any_S --keyframe {HWACCEL}")
