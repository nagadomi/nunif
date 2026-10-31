import os
from typing import NamedTuple

import pytest

from nunif.cli.diff_video import compare_videos
from tests.helpers.race_condition_data import ensure_race_condition_data
from tests.helpers.runner import run

COMMON_REF_ARGS = ["--low-vram", "--scene-detect", "--ema-normalize", "--ema-buffer", "30", "-y"]
COMMON_TEST_ARGS = [
    "--batch-size",
    "4",
    "--max-workers",
    "2",
    "--cuda-stream",
    "--scene-detect",
    "--ema-normalize",
    "--ema-buffer",
    "30",
    "-y",
]

MIN_PSNR = 44.0


class RaceConditionSetup(NamedTuple):
    work_dir: str
    sd: str
    hdr: str
    sd_noaudio: str
    hdr_noaudio: str
    rgba: str


@pytest.fixture(scope="module")
def race_condition_setup(tmp_path_factory: pytest.TempPathFactory) -> RaceConditionSetup:
    tmp_dir = tmp_path_factory.mktemp("race_condition")
    data_dir = str(tmp_dir / "data")
    work_dir = str(tmp_dir / "work")
    os.makedirs(work_dir, exist_ok=True)
    ensure_race_condition_data(base_dir=data_dir)
    return RaceConditionSetup(
        work_dir=work_dir,
        sd=os.path.join(data_dir, "sd.mkv"),
        hdr=os.path.join(data_dir, "hdr.mkv"),
        sd_noaudio=os.path.join(data_dir, "sd_noaudio.mkv"),
        hdr_noaudio=os.path.join(data_dir, "hdr_noaudio.mkv"),
        rgba=os.path.join(data_dir, "rgba.mkv"),
    )


def run_and_compare(work_dir: str, test_name: str, input_video: str, extra_args: list[str]) -> bool:
    ref_output = os.path.join(work_dir, f"{test_name}_ref.mkv")
    test_output = os.path.join(work_dir, f"{test_name}_test.mkv")

    # Run reference (sequential, low-vram)
    run(["python", "-m", "iw3.cli", "-i", input_video, "-o", ref_output, *COMMON_REF_ARGS, *extra_args])

    # Run test target (batch, multi-worker, cuda-stream)
    run(["python", "-m", "iw3.cli", "-i", input_video, "-o", test_output, *COMMON_TEST_ARGS, *extra_args])

    # Compare with PSNR
    result = compare_videos(ref_output, test_output)
    if result.identical:
        print(f"[PASS] {test_name}: PSNR inf (Bit-identical)")
        return True
    elif result.mean_psnr >= MIN_PSNR:
        print(f"[PASS] {test_name}: PSNR {result.mean_psnr:.2f} >= {MIN_PSNR}")
        return True
    else:
        print(f"[FAIL] {test_name}: PSNR {result.mean_psnr:.2f} < {MIN_PSNR}")
        return False


@pytest.mark.slow
def test_vda_s_sd_libx265(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_sd_libx265",
        race_condition_setup.sd,
        ["--depth-model", "VDA_S", "--video-codec", "libx265"],
    )


@pytest.mark.slow
def test_vda_s_sd_nvenc(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_sd_nvenc",
        race_condition_setup.sd,
        ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"],
    )


@pytest.mark.slow
def test_vda_s_hdr_libx265(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_hdr_libx265",
        race_condition_setup.hdr,
        ["--depth-model", "VDA_S", "--video-codec", "libx265", "--pix-fmt", "yuv420p10le"],
    )


@pytest.mark.slow
def test_any_s_sd_libx265(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_sd_libx265",
        race_condition_setup.sd,
        ["--depth-model", "Any_S", "--video-codec", "libx265"],
    )


@pytest.mark.slow
def test_any_s_sd_nvenc(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_sd_nvenc",
        race_condition_setup.sd,
        ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"],
    )


@pytest.mark.slow
def test_any_s_hdr_nvenc(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_hdr_nvenc",
        race_condition_setup.hdr,
        ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc", "--pix-fmt", "yuv420p10le"],
    )


@pytest.mark.slow
def test_any_s_sd_inpaint(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_sd_inpaint",
        race_condition_setup.sd,
        ["--depth-model", "Any_S", "--method", "forward_inpaint", "--video-codec", "libx265"],
    )


@pytest.mark.slow
def test_vda_s_sd_nvenc_noaudio(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_sd_nvenc_noaudio",
        race_condition_setup.sd_noaudio,
        ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"],
    )


@pytest.mark.slow
def test_any_s_sd_nvenc_noaudio(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_sd_nvenc_noaudio",
        race_condition_setup.sd_noaudio,
        ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"],
    )


@pytest.mark.slow
def test_vda_s_sd_libx265_noaudio(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_sd_libx265_noaudio",
        race_condition_setup.sd_noaudio,
        ["--depth-model", "VDA_S", "--video-codec", "libx265"],
    )


@pytest.mark.slow
def test_any_s_rgba_ffv1(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "any_s_rgba_ffv1",
        race_condition_setup.rgba,
        ["--depth-model", "Any_S", "--video-codec", "ffv1", "--pix-fmt", "yuva420p"],
    )


@pytest.mark.slow
def test_vda_s_rgba_ffv1(race_condition_setup: RaceConditionSetup) -> None:
    assert run_and_compare(
        race_condition_setup.work_dir,
        "vda_s_rgba_ffv1",
        race_condition_setup.rgba,
        ["--depth-model", "VDA_S", "--video-codec", "ffv1", "--pix-fmt", "yuva420p"],
    )
