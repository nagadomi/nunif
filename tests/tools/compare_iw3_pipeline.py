import argparse
import os
import shlex
import shutil

from nunif.cli.diff_video import compare_videos
from tests.helpers.git_worktree import resolve_repo_path
from tests.helpers.runner import run

DEFAULT_BASELINE = "9fd1812477a2a7f372c87ac7ac14fdfab0a7c3ed"
DEFAULT_TARGET = "."


def run_pipeline_for_repo(
    repo_path: str,
    output_dir: str,
    cache_dir: str,
    video_path: str,
    hwaccel: bool,
    common_options: str,
) -> None:
    req_file = os.path.join(repo_path, "requirements.txt")
    if os.path.exists(req_file):
        run(f"python -m pip install -r {shlex.quote(req_file)}")

    extra_args = "--hwaccel cuda" if hwaccel else ""
    cache_arg = shlex.quote(os.path.abspath(cache_dir))

    def out_path(filename: str) -> str:
        return shlex.quote(os.path.abspath(os.path.join(output_dir, filename)))

    def run_iw3(cmd_args: str) -> None:
        base = f"python -m iw3.cli -y -i {shlex.quote(video_path)} {common_options}"
        if extra_args:
            base = f"{base} {extra_args}"
        run(f"{base} {cmd_args}", cwd=repo_path)

    run_iw3(f"-o {out_path('batch.mkv')} --depth-model Any_V2_S --max-workers 2 --batch-size 2")
    run_iw3(
        f"-o {out_path('batch_forward_fill.mkv')} --depth-model Any_V2_S "
        f"--method forward_fill --max-workers 1 --batch-size 2"
    )
    run_iw3(
        f"-o {out_path('batch_yuv420p10le.mkv')} --depth-model Any_V2_S "
        f"--max-workers 2 --batch-size 2 --pix-fmt yuv420p10le"
    )
    run_iw3(f"-o {out_path('batch_cuda.mkv')} --depth-model Any_V2_S --max-workers 2 --batch-size 2 --cuda-stream")
    run_iw3(f"-o {out_path('low_vram.mkv')} --depth-model Any_V2_S --low-vram")
    run_iw3(f"-o {out_path('low_vram_yuv420p10le.mkv')} --depth-model Any_V2_S --low-vram --pix-fmt yuv420p10le")
    run_iw3(
        f"-o {out_path('batch_ema.mkv')} --depth-model Any_V2_S "
        f"--max-workers 2 --batch-size 2 --cuda-stream --ema-normalize"
    )
    run_iw3(f"-o {out_path('low_vram_ema.mkv')} --depth-model Any_V2_S --low-vram --ema-normalize")
    run_iw3(
        f"-o {out_path('vda.mkv')} --depth-model VDA_S --ema-normalize "
        f"--batch-size 2 --scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('vda_yuv420p10le.mkv')} --depth-model VDA_S --ema-normalize "
        f"--batch-size 2 --scene-detect --scene-cache-dir {cache_arg} --pix-fmt yuv420p10le"
    )
    run_iw3(
        f"-o {out_path('vda_stream.mkv')} --depth-model VDA_Stream_S --ema-normalize "
        f"--batch-size 2 --scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('vda_stream_yuv420p10le.mkv')} --depth-model VDA_Stream_S --ema-normalize "
        f"--batch-size 2 --scene-detect --scene-cache-dir {cache_arg} --pix-fmt yuv420p10le"
    )
    run_iw3(
        f"-o {out_path('inpaint_batch.mkv')} --depth-model VDA_S --method mlbw_l2_inpaint "
        f"--inpaint-max-width 1920 --ema-normalize --ema-buffer 30 --batch-size 2 "
        f"--scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('inpaint_vda.mkv')} --depth-model VDA_S --method mlbw_l2_inpaint "
        f"--inpaint-max-width 1920 --ema-normalize --ema-buffer 30 --batch-size 2 "
        f"--scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('inpaint_vda_stream.mkv')} "
        f"--depth-model VDA_Stream_S --method mlbw_l2_inpaint "
        f"--inpaint-max-width 1920 --ema-normalize --ema-buffer 30 --batch-size 2 "
        f"--scene-detect --scene-cache-dir {cache_arg}"
    )

    # export
    run_iw3(
        f"-o {out_path('export')} --export --depth-model Any_V2_S --ema-normalize "
        f"--ema-buffer 30 --batch-size 2 --scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('export_depth_only')} --export-disparity --export-depth-only "
        f"--depth-model Any_V2_S --ema-normalize --ema-buffer 30 --batch-size 2 "
        f"--scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('export_vda')} --export --depth-model VDA_S --ema-normalize "
        f"--ema-buffer 30 --batch-size 2 --scene-detect --scene-cache-dir {cache_arg}"
    )
    run_iw3(
        f"-o {out_path('export_vda_depth_only')} --export-disparity --export-depth-only "
        f"--depth-model VDA_S --ema-normalize --ema-buffer 30 --batch-size 2 "
        f"--scene-detect --scene-cache-dir {cache_arg}"
    )


def compare_outputs(ref_dir: str, target_dir: str, video_path: str) -> None:
    output_files = [
        "batch.mkv",
        "batch_forward_fill.mkv",
        "batch_yuv420p10le.mkv",
        "batch_cuda.mkv",
        "low_vram.mkv",
        "low_vram_yuv420p10le.mkv",
        "batch_ema.mkv",
        "low_vram_ema.mkv",
        "vda.mkv",
        "vda_yuv420p10le.mkv",
        "vda_stream.mkv",
        "vda_stream_yuv420p10le.mkv",
        "inpaint_batch.mkv",
        "inpaint_vda.mkv",
        "inpaint_vda_stream.mkv",
    ]

    for filename in output_files:
        file_ref = os.path.join(ref_dir, filename)
        file_target = os.path.join(target_dir, filename)

        print(f"--- Comparing: {filename} ---")
        if not os.path.exists(file_ref) or not os.path.exists(file_target):
            print("  [!!] Error: Output file missing.")
            continue

        res = compare_videos(file_ref, file_target)
        if res.identical or res.mean_psnr >= 38.0:
            print(f"  [OK] Quality Pass: PSNR {res.mean_psnr}")
        else:
            print(f"  [FAIL] Quality Low: PSNR {res.mean_psnr}")

    output_dirs = [
        "export",
        "export_depth_only",
        "export_vda",
        "export_vda_depth_only",
    ]
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    for d in output_dirs:
        dir_ref = os.path.join(ref_dir, d, video_basename, "depth")
        dir_target = os.path.join(target_dir, d, video_basename, "depth")
        run(f"python -m nunif.cli.diff_image -i {shlex.quote(dir_ref)} {shlex.quote(dir_target)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare iw3 pipeline outputs against a baseline git revision or directory"
    )
    parser.add_argument("video", type=str, help="Path to test input video")
    parser.add_argument(
        "--ref",
        type=str,
        default=DEFAULT_BASELINE,
        help=f"Baseline git commit/ref or directory to compare against (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Target git commit/ref or directory to test (default: '{DEFAULT_TARGET}')",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)

    ref_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "ref")
    target_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "target")
    cache_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "cache")
    common_options = "--video-codec hevc_nvenc"

    shutil.rmtree(ref_dir, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print("iw3 Pipeline Comparison")
    print(f"Baseline (ref) : {args.ref}")
    print(f"Target         : {args.target}")
    print(f"Input Video    : {video_path}")
    print("=" * 70)

    try:
        with resolve_repo_path(args.ref) as ref_path:
            run_pipeline_for_repo(
                repo_path=ref_path,
                output_dir=ref_dir,
                cache_dir=cache_dir,
                video_path=video_path,
                hwaccel=False,
                common_options=common_options,
            )

        with resolve_repo_path(args.target) as target_path:
            run_pipeline_for_repo(
                repo_path=target_path,
                output_dir=target_dir,
                cache_dir=cache_dir,
                video_path=video_path,
                hwaccel=True,
                common_options=common_options,
            )

        compare_outputs(ref_dir, target_dir, video_path)

    finally:
        orig_req = "requirements.txt"
        if os.path.exists(orig_req):
            run(f"python -m pip install -r {orig_req}")


if __name__ == "__main__":
    main()
