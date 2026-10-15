import argparse
import os
import shutil

from nunif.cli.diff_video import VideoMetadata, compare_videos, get_video_metadata
from tests.helpers.git_worktree import resolve_repo_path
from tests.helpers.runner import run

DEFAULT_BASELINE = "9fd1812477a2a7f372c87ac7ac14fdfab0a7c3ed"
DEFAULT_TARGET = "."


def run_color_transform_for_repo(
    repo_path: str,
    out_dir: str,
    video_path: str,
    is_target: bool,
) -> None:
    req_file = os.path.join(repo_path, "requirements.txt")
    if os.path.exists(req_file):
        run(["python", "-m", "pip", "install", "-r", req_file])

    iw3_options = [
        "-i",
        video_path,
        "--yes",
        "--half-sbs",
        "--method",
        "row_flow_v3",
        "--depth-model",
        "Any_V2_S",
        "--batch-size",
        "2",
        "--max-workers",
        "2",
        "--cuda-stream",
    ]

    for encoder in ["libx265", "hevc_nvenc"]:
        for pix_fmt in ["yuv420p", "yuv420p10le"]:
            for colorspace in ["auto", "bt709-tv", "bt709-pc", "bt601-tv"]:
                if encoder == "hevc_nvenc" and colorspace == "bt709-pc":
                    continue

                out_file = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                out_file_arg = os.path.abspath(out_file)
                print("-" * 50)
                print(f"Processing: {out_file}")
                run(
                    [
                        "python",
                        "-m",
                        "iw3.cli",
                        *iw3_options,
                        "--video-codec",
                        encoder,
                        "--pix-fmt",
                        pix_fmt,
                        "--colorspace",
                        colorspace,
                        "-o",
                        out_file_arg,
                    ],
                    cwd=repo_path,
                )

                out_file_vf = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                out_file_vf_arg = os.path.abspath(out_file_vf)
                run(
                    [
                        "python",
                        "-m",
                        "iw3.cli",
                        *iw3_options,
                        "--video-codec",
                        encoder,
                        "--pix-fmt",
                        pix_fmt,
                        "--colorspace",
                        colorspace,
                        "--vf",
                        "crop=x=0:y=0:w=iw:h=ih",
                        "-o",
                        out_file_vf_arg,
                    ],
                    cwd=repo_path,
                )

                if is_target:
                    out_file_cuda = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda.mkv")
                    out_file_cuda_arg = os.path.abspath(out_file_cuda)
                    print(f"Processing: {out_file_cuda}")
                    run(
                        [
                            "python",
                            "-m",
                            "iw3.cli",
                            *iw3_options,
                            "--video-codec",
                            encoder,
                            "--pix-fmt",
                            pix_fmt,
                            "--colorspace",
                            colorspace,
                            "--hwaccel",
                            "cuda",
                            "-o",
                            out_file_cuda_arg,
                        ],
                        cwd=repo_path,
                    )

                    out_file_cuda_vf = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda_vf.mkv")
                    out_file_cuda_vf_arg = os.path.abspath(out_file_cuda_vf)
                    print(f"Processing: {out_file_cuda_vf}")
                    run(
                        [
                            "python",
                            "-m",
                            "iw3.cli",
                            *iw3_options,
                            "--video-codec",
                            encoder,
                            "--pix-fmt",
                            pix_fmt,
                            "--colorspace",
                            colorspace,
                            "--hwaccel",
                            "cuda",
                            "--vf",
                            "crop=x=0:y=0:w=iw:h=ih",
                            "-o",
                            out_file_cuda_vf_arg,
                        ],
                        cwd=repo_path,
                    )


def compare_color_outputs(ref_dir: str, target_dir: str) -> None:
    for encoder in ["libx265", "hevc_nvenc"]:
        for pix_fmt in ["yuv420p", "yuv420p10le"]:
            for colorspace in ["auto", "bt709-tv", "bt709-pc", "bt601-tv"]:
                out_ref = os.path.join(ref_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                out_target = os.path.join(target_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                out_target_cuda = os.path.join(target_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda.mkv")

                out_ref_vf = os.path.join(ref_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                out_target_vf = os.path.join(target_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                out_target_cuda_vf = os.path.join(target_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda_vf.mkv")

                if not os.path.exists(out_ref) or not os.path.exists(out_target):
                    print(f"Skip: File not found ({encoder} {pix_fmt} {colorspace})")
                    continue

                print("-" * 50)
                print(f"Comparing: {encoder} / {pix_fmt} / {colorspace}")

                m_ref = get_video_metadata(out_ref)
                m_target = get_video_metadata(out_target)
                m_target_cuda = get_video_metadata(out_target_cuda) if os.path.exists(out_target_cuda) else None

                m_ref_vf = get_video_metadata(out_ref_vf)
                m_target_vf = get_video_metadata(out_target_vf)
                m_target_cuda_vf = (
                    get_video_metadata(out_target_cuda_vf) if os.path.exists(out_target_cuda_vf) else None
                )

                def meta_str(m: VideoMetadata) -> str:
                    return f"cs={m.colorspace},trc={m.color_trc},pri={m.color_primaries},range={m.color_range}"

                if meta_str(m_ref) == meta_str(m_target):
                    print(f"  [OK] Metadata Match: {meta_str(m_ref)}")
                else:
                    print("  [NG] Metadata Mismatch!")
                    print(f"       Ref   : {meta_str(m_ref)}")
                    print(f"       Target: {meta_str(m_target)}")

                if m_target_cuda is not None:
                    if meta_str(m_ref) == meta_str(m_target_cuda):
                        print(f"  [OK] Metadata Match (CUDA): {meta_str(m_ref)}")
                    else:
                        print("  [NG] Metadata Mismatch! (CUDA)")
                        print(f"       Ref   : {meta_str(m_ref)}")
                        print(f"       Target: {meta_str(m_target_cuda)}")

                if meta_str(m_ref_vf) == meta_str(m_target_vf):
                    print(f"  [OK] Metadata Match (vf): {meta_str(m_ref_vf)}")
                else:
                    print("  [NG] Metadata Mismatch! (vf)")
                    print(f"       Ref   : {meta_str(m_ref_vf)}")
                    print(f"       Target: {meta_str(m_target_vf)}")

                if m_target_cuda_vf is not None:
                    if meta_str(m_ref_vf) == meta_str(m_target_cuda_vf):
                        print(f"  [OK] Metadata Match (CUDA vf): {meta_str(m_ref_vf)}")
                    else:
                        print("  [NG] Metadata Mismatch! (CUDA vf)")
                        print(f"       Ref   : {meta_str(m_ref_vf)}")
                        print(f"       Target: {meta_str(m_target_cuda_vf)}")

                # PSNR checks
                pairs = [
                    (out_ref, out_target, "Quality Pass"),
                    (out_ref_vf, out_target_vf, "Quality Pass (vf)"),
                ]
                if os.path.exists(out_target_cuda):
                    pairs.append((out_ref, out_target_cuda, "Quality Pass (CUDA)"))
                if os.path.exists(out_target_cuda_vf):
                    pairs.append((out_ref_vf, out_target_cuda_vf, "Quality Pass (CUDA vf)"))

                for p1, p2, label in pairs:
                    res = compare_videos(p1, p2)
                    if res.identical or res.mean_psnr >= 38.0:
                        print(f"  [OK] {label}: PSNR {res.mean_psnr}")
                    else:
                        print(f"  [FAIL] Quality Low ({label}): PSNR {res.mean_psnr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare color transform outputs against a baseline git revision or directory"
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

    ref_dir = os.path.join("tests", "data", "compare_color_transform", "ref")
    target_dir = os.path.join("tests", "data", "compare_color_transform", "target")

    shutil.rmtree(ref_dir, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print("Color Transform Comparison")
    print(f"Baseline (ref) : {args.ref}")
    print(f"Target         : {args.target}")
    print(f"Input Video    : {video_path}")
    print("=" * 70)

    try:
        with resolve_repo_path(args.ref) as ref_path:
            run_color_transform_for_repo(ref_path, ref_dir, video_path, is_target=False)

        with resolve_repo_path(args.target) as target_path:
            run_color_transform_for_repo(target_path, target_dir, video_path, is_target=True)

        compare_color_outputs(ref_dir, target_dir)

    finally:
        orig_req = "requirements.txt"
        if os.path.exists(orig_req):
            run(["python", "-m", "pip", "install", "-r", orig_req])


if __name__ == "__main__":
    main()
