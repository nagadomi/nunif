import os

import pytest

from nunif.cli.diff_video import compare_videos
from tests.helpers.runner import run

WORK_DIR = "tests/data/race_condition_test"

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

TEST_CASES = [
    ("vda_s_sd_libx265", "sd.mkv", ["--depth-model", "VDA_S", "--video-codec", "libx265"], False),
    ("vda_s_sd_nvenc", "sd.mkv", ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"], True),
    (
        "vda_s_hdr_libx265",
        "hdr.mkv",
        ["--depth-model", "VDA_S", "--video-codec", "libx265", "--pix-fmt", "yuv420p10le"],
        False,
    ),
    ("any_s_sd_libx265", "sd.mkv", ["--depth-model", "Any_S", "--video-codec", "libx265"], False),
    ("any_s_sd_nvenc", "sd.mkv", ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"], True),
    (
        "any_s_hdr_nvenc",
        "hdr.mkv",
        ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc", "--pix-fmt", "yuv420p10le"],
        True,
    ),
    (
        "any_s_sd_inpaint",
        "sd.mkv",
        ["--depth-model", "Any_S", "--method", "forward_inpaint", "--video-codec", "libx265"],
        False,
    ),
    ("vda_s_sd_nvenc_noaudio", "sd_noaudio.mkv", ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"], True),
    ("any_s_sd_nvenc_noaudio", "sd_noaudio.mkv", ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"], True),
    ("vda_s_sd_libx265_noaudio", "sd_noaudio.mkv", ["--depth-model", "VDA_S", "--video-codec", "libx265"], False),
]


def run_and_compare(test_name: str, input_video: str, extra_args: list[str]) -> bool:
    os.makedirs(WORK_DIR, exist_ok=True)
    ref_output = os.path.join(WORK_DIR, f"{test_name}_ref.mkv")
    test_output = os.path.join(WORK_DIR, f"{test_name}_test.mkv")

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
@pytest.mark.parametrize(
    "name,video_file,extra_args,is_hwaccel",
    TEST_CASES,
)
def test_race_condition(
    race_condition_data_dir: str,
    name: str,
    video_file: str,
    extra_args: list[str],
    is_hwaccel: bool,
) -> None:
    input_video = os.path.join(race_condition_data_dir, video_file)
    assert run_and_compare(name, input_video, extra_args), f"Race condition test failed for {name}"
