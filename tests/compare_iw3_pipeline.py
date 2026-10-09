import argparse
import os
import shlex
import shutil

from nunif.cli.diff_video import compare_videos
from tests.shell_utils import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare iw3 pipeline between two nunif repository versions")
    parser.add_argument("repo1", type=str, help="Path to repo1")
    parser.add_argument("repo2", type=str, help="Path to repo2")
    parser.add_argument("video", type=str, help="Path to test video")
    args = parser.parse_args()

    repo1_path = os.path.abspath(args.repo1)
    repo2_path = os.path.abspath(args.repo2)
    video_path = os.path.abspath(args.video)

    output1_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "repo1")
    output2_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "repo2")
    cache_dir = os.path.join("tests", "data", "compare_iw3_pipeline", "cache")

    common_options = "--video-codec hevc_nvenc"

    shutil.rmtree(output1_dir, ignore_errors=True)
    shutil.rmtree(output2_dir, ignore_errors=True)
    os.makedirs(output1_dir, exist_ok=True)
    os.makedirs(output2_dir, exist_ok=True)

    repos = [repo1_path, repo2_path]
    outputs = [output1_dir, output2_dir]

    try:
        for i in range(len(repos)):
            repo = repos[i]
            out_dir = outputs[i]

            extra_args = "--hwaccel cuda" if i == 1 else ""

            req_file = os.path.join(repo, "requirements.txt")
            if os.path.exists(req_file):
                run(f"python -m pip install -r {shlex.quote(req_file)}")

            cache_arg = shlex.quote(os.path.abspath(cache_dir))

            def out_path(filename: str) -> str:
                return shlex.quote(os.path.abspath(os.path.join(out_dir, filename)))

            def run_iw3(cmd_args: str) -> None:
                base = f"python -m iw3.cli -y -i {shlex.quote(video_path)} {common_options}"
                if extra_args:
                    base = f"{base} {extra_args}"
                run(f"{base} {cmd_args}", cwd=repo)

            run_iw3(f"-o {out_path('batch.mkv')} --depth-model Any_V2_S --max-workers 2 --batch-size 2")
            run_iw3(
                f"-o {out_path('batch_forward_fill.mkv')} --depth-model Any_V2_S "
                f"--method forward_fill --max-workers 1 --batch-size 2"
            )
            run_iw3(
                f"-o {out_path('batch_yuv420p10le.mkv')} --depth-model Any_V2_S "
                f"--max-workers 2 --batch-size 2 --pix-fmt yuv420p10le"
            )
            run_iw3(
                f"-o {out_path('batch_cuda.mkv')} --depth-model Any_V2_S --max-workers 2 --batch-size 2 --cuda-stream"
            )
            run_iw3(f"-o {out_path('low_vram.mkv')} --depth-model Any_V2_S --low-vram")
            run_iw3(
                f"-o {out_path('low_vram_yuv420p10le.mkv')} --depth-model Any_V2_S --low-vram --pix-fmt yuv420p10le"
            )
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
            file1 = os.path.join(output1_dir, filename)
            file2 = os.path.join(output2_dir, filename)

            print(f"--- Comparing: {filename} ---")
            if not os.path.exists(file1) or not os.path.exists(file2):
                print("  [!!] Error: Output file missing.")
                continue

            res = compare_videos(file1, file2)
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
            dir1 = os.path.join(output1_dir, d, video_basename, "depth")
            dir2 = os.path.join(output2_dir, d, video_basename, "depth")
            run(f"python -m nunif.cli.diff_image -i {shlex.quote(dir1)} {shlex.quote(dir2)}")

    finally:
        orig_req = "requirements.txt"
        if os.path.exists(orig_req):
            run(f"python -m pip install -r {orig_req}")


if __name__ == "__main__":
    main()
