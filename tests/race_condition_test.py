import os
import shutil
import sys

from nunif.cli.diff_video import compare_videos
from tests.shell_utils import run, run_module

WORK_DIR = "tests/data/race_condition_test"
SD_VIDEO = "tests/data/race_condition/sd.mkv"
HDR_VIDEO = "tests/data/race_condition/hdr.mkv"
SD_NOAUDIO_VIDEO = "tests/data/race_condition/sd_noaudio.mkv"
HDR_NOAUDIO_VIDEO = "tests/data/race_condition/hdr_noaudio.mkv"

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


def run_and_compare(test_name: str, input_video: str, extra_args: list[str]) -> bool:
    print("=" * 70)
    print(f"Running Test: {test_name}")
    print(f"Input: {input_video}")
    print(f"Extra args: {' '.join(extra_args)}")
    print("=" * 70)

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
    elif result.mean_psnr >= 38.0:
        print(f"[PASS] {test_name}: PSNR {result.mean_psnr} >= 38")
        return True
    else:
        print(f"[FAIL] {test_name}: PSNR {result.mean_psnr} < 38")
        return False


def main() -> None:
    # Generate test data if not exists
    run_module("tests.race_condition_data")

    shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    failed = False
    tests = [
        ("vda_s_sd_libx265", SD_VIDEO, ["--depth-model", "VDA_S", "--video-codec", "libx265"]),
        ("vda_s_sd_nvenc", SD_VIDEO, ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"]),
        (
            "vda_s_hdr_libx265",
            HDR_VIDEO,
            ["--depth-model", "VDA_S", "--video-codec", "libx265", "--pix-fmt", "yuv420p10le"],
        ),
        ("any_s_sd_libx265", SD_VIDEO, ["--depth-model", "Any_S", "--video-codec", "libx265"]),
        ("any_s_sd_nvenc", SD_VIDEO, ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"]),
        (
            "any_s_hdr_nvenc",
            HDR_VIDEO,
            ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc", "--pix-fmt", "yuv420p10le"],
        ),
        (
            "any_s_sd_inpaint",
            SD_VIDEO,
            ["--depth-model", "Any_S", "--method", "forward_inpaint", "--video-codec", "libx265"],
        ),
        ("vda_s_sd_nvenc_noaudio", SD_NOAUDIO_VIDEO, ["--depth-model", "VDA_S", "--video-codec", "hevc_nvenc"]),
        ("any_s_sd_nvenc_noaudio", SD_NOAUDIO_VIDEO, ["--depth-model", "Any_S", "--video-codec", "hevc_nvenc"]),
        ("vda_s_sd_libx265_noaudio", SD_NOAUDIO_VIDEO, ["--depth-model", "VDA_S", "--video-codec", "libx265"]),
    ]

    for name, video, args in tests:
        if not run_and_compare(name, video, args):
            failed = True

    if not failed:
        print("=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        sys.exit(0)
    else:
        print("=" * 70)
        print(f"SOME TESTS FAILED! (Check files in {WORK_DIR})")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
