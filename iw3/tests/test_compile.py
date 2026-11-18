import os
import sys
from typing import NamedTuple

import pytest
import torch

from nunif.cli.diff_video import compare_videos
from nunif.utils.video.utils import is_nvidia_gpu
from tests.helpers.race_condition_data import ensure_race_condition_data
from tests.helpers.runner import run

NVIDIA_ONLY = True


def _is_compile_supported() -> bool:
    if sys.platform != "linux":
        return False
    if NVIDIA_ONLY:
        return torch.cuda.is_available() and is_nvidia_gpu(torch.device("cuda:0"))
    return True


pytestmark = pytest.mark.skipif(
    not _is_compile_supported(),
    reason="torch.compile tests are only run on Linux with NVIDIA GPU",
)

COMMON_ARGS = [
    "--batch-size",
    "4",
    "--max-workers",
    "2",
    "--cuda-stream",
    "--scene-detect",
    "--ema-normalize",
    "--ema-buffer",
    "10",
    "--video-codec",
    "libx265",
    "--crf",
    "4",
    "-y",
]

MIN_PSNR = 44.0


class CompileTestSetup(NamedTuple):
    work_dir: str
    sd: str


@pytest.fixture(scope="module")
def compile_test_setup(tmp_path_factory: pytest.TempPathFactory) -> CompileTestSetup:
    tmp_dir = tmp_path_factory.mktemp("compile")
    data_dir = str(tmp_dir / "data")
    work_dir = str(tmp_dir / "work")
    os.makedirs(work_dir, exist_ok=True)
    ensure_race_condition_data(base_dir=data_dir)
    return CompileTestSetup(
        work_dir=work_dir,
        sd=os.path.join(data_dir, "sd.mkv"),
    )


def run_and_compare(work_dir: str, test_name: str, input_video: str, extra_args: list[str]) -> bool:
    ref_output = os.path.join(work_dir, f"{test_name}_ref.mkv")
    test_output = os.path.join(work_dir, f"{test_name}_compile.mkv")

    run(["python", "-m", "iw3.cli", "-i", input_video, "-o", ref_output, *COMMON_ARGS, *extra_args])
    run(["python", "-m", "iw3.cli", "-i", input_video, "-o", test_output, "--compile", *COMMON_ARGS, *extra_args])

    result = compare_videos(ref_output, test_output)
    if result.identical:
        print(f"[PASS] {test_name}: PSNR inf (Bit-identical)")
        return True
    elif result.mean_psnr >= MIN_PSNR:
        print(
            f"[PASS] {test_name}: Mean PSNR {result.mean_psnr:.2f} dB "
            f"(Min: {result.min_psnr:.2f} dB, Max: {result.max_psnr:.2f} dB) >= {MIN_PSNR}"
        )
        return True
    else:
        print(
            f"[FAIL] {test_name}: Mean PSNR {result.mean_psnr:.2f} dB "
            f"(Min: {result.min_psnr:.2f} dB, Max: {result.max_psnr:.2f} dB) < {MIN_PSNR}"
        )
        return False


@pytest.mark.slow
def test_compile_any_s(compile_test_setup: CompileTestSetup) -> None:
    assert run_and_compare(
        compile_test_setup.work_dir,
        "compile_any_s",
        compile_test_setup.sd,
        ["--depth-model", "Any_S"],
    )


@pytest.mark.slow
def test_compile_vda_s(compile_test_setup: CompileTestSetup) -> None:
    assert run_and_compare(
        compile_test_setup.work_dir,
        "compile_vda_s",
        compile_test_setup.sd,
        ["--depth-model", "VDA_S"],
    )


@pytest.mark.slow
def test_compile_any_s_inpaint(compile_test_setup: CompileTestSetup) -> None:
    assert run_and_compare(
        compile_test_setup.work_dir,
        "compile_any_s_inpaint",
        compile_test_setup.sd,
        ["--depth-model", "Any_S", "--method", "forward_inpaint"],
    )


@pytest.mark.slow
def test_compile_any_s_tta_sod(compile_test_setup: CompileTestSetup) -> None:
    assert run_and_compare(
        compile_test_setup.work_dir,
        "compile_any_s_tta_sod",
        compile_test_setup.sd,
        ["--depth-model", "Any_S", "--tta", "--depth-aa", "--convergence-mode", "sod_v1"],
    )


@pytest.mark.slow
def test_compile_any_s_low_vram(compile_test_setup: CompileTestSetup) -> None:
    assert run_and_compare(
        compile_test_setup.work_dir,
        "compile_any_s_low_vram",
        compile_test_setup.sd,
        ["--depth-model", "Any_S", "--low-vram"],
    )
