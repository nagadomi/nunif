import argparse
import os
import shlex

from nunif.cli.diff_video import VideoMetadata, compare_videos, get_video_metadata
from tests.shell_utils import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare color transform between two nunif repository versions")
    parser.add_argument("repo1", type=str, help="Path to repo1")
    parser.add_argument("repo2", type=str, help="Path to repo2")
    parser.add_argument("video", type=str, help="Path to test video")
    args = parser.parse_args()

    repo1_path = os.path.abspath(args.repo1)
    repo2_path = os.path.abspath(args.repo2)
    video_path = os.path.abspath(args.video)

    output1_dir = os.path.join("tests", "data", "compare_color_transform", "repo1")
    output2_dir = os.path.join("tests", "data", "compare_color_transform", "repo2")

    iw3_options = (
        f"-i {shlex.quote(video_path)} --yes --half-sbs --method row_flow_v3 "
        f"--depth-model Any_V2_S --batch-size 2 --max-workers 2 --cuda-stream"
    )

    os.makedirs(output1_dir, exist_ok=True)
    os.makedirs(output2_dir, exist_ok=True)

    repos = [repo1_path, repo2_path]
    outputs = [output1_dir, output2_dir]

    try:
        for idx in range(len(repos)):
            repo = repos[idx]
            out_dir = outputs[idx]

            req_file = os.path.join(repo, "requirements.txt")
            if os.path.exists(req_file):
                run(f"python -m pip install -r {shlex.quote(req_file)}")

            for encoder in ["libx265", "hevc_nvenc"]:
                for pix_fmt in ["yuv420p", "yuv420p10le"]:
                    for colorspace in ["auto", "bt709-tv", "bt709-pc", "bt601-tv"]:
                        if encoder == "hevc_nvenc" and colorspace == "bt709-pc":
                            continue

                        out_file = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                        out_file_arg = shlex.quote(os.path.abspath(out_file))
                        print("-" * 50)
                        print(f"Processing: {out_file}")
                        run(
                            f"python -m iw3.cli {iw3_options} --video-codec {encoder} --pix-fmt {pix_fmt} "
                            f"--colorspace {colorspace} -o {out_file_arg}",
                            cwd=repo,
                        )

                        out_file_vf = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                        out_file_vf_arg = shlex.quote(os.path.abspath(out_file_vf))
                        run(
                            f"python -m iw3.cli {iw3_options} --video-codec {encoder} --pix-fmt {pix_fmt} "
                            f"--colorspace {colorspace} --vf crop=x=0:y=0:w=iw:h=ih -o {out_file_vf_arg}",
                            cwd=repo,
                        )

                        if repo == repo2_path:
                            out_file_cuda = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda.mkv")
                            out_file_cuda_arg = shlex.quote(os.path.abspath(out_file_cuda))
                            print(f"Processing: {out_file_cuda}")
                            run(
                                f"python -m iw3.cli {iw3_options} --video-codec {encoder} --pix-fmt {pix_fmt} "
                                f"--colorspace {colorspace} --hwaccel cuda -o {out_file_cuda_arg}",
                                cwd=repo,
                            )

                            out_file_cuda_vf = os.path.join(out_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda_vf.mkv")
                            out_file_cuda_vf_arg = shlex.quote(os.path.abspath(out_file_cuda_vf))
                            print(f"Processing: {out_file_cuda_vf}")
                            run(
                                f"python -m iw3.cli {iw3_options} --video-codec {encoder} --pix-fmt {pix_fmt} "
                                f"--colorspace {colorspace} --hwaccel cuda "
                                f"--vf crop=x=0:y=0:w=iw:h=ih -o {out_file_cuda_vf_arg}",
                                cwd=repo,
                            )

        # Compare outputs
        for encoder in ["libx265", "hevc_nvenc"]:
            for pix_fmt in ["yuv420p", "yuv420p10le"]:
                for colorspace in ["auto", "bt709-tv", "bt709-pc", "bt601-tv"]:
                    out1 = os.path.join(output1_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                    out2 = os.path.join(output2_dir, f"{encoder}_{pix_fmt}_{colorspace}.mkv")
                    out3 = os.path.join(output2_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda.mkv")

                    out1_vf = os.path.join(output1_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                    out2_vf = os.path.join(output2_dir, f"{encoder}_{pix_fmt}_{colorspace}_vf.mkv")
                    out3_vf = os.path.join(output2_dir, f"{encoder}_{pix_fmt}_{colorspace}_cuda_vf.mkv")

                    if not os.path.exists(out1) or not os.path.exists(out2):
                        print(f"Skip: File not found ({encoder} {pix_fmt} {colorspace})")
                        continue

                    print("-" * 50)
                    print(f"Comparing: {encoder} / {pix_fmt} / {colorspace}")

                    m1 = get_video_metadata(out1)
                    m2 = get_video_metadata(out2)
                    m3 = get_video_metadata(out3) if os.path.exists(out3) else None

                    m1_vf = get_video_metadata(out1_vf)
                    m2_vf = get_video_metadata(out2_vf)
                    m3_vf = get_video_metadata(out3_vf) if os.path.exists(out3_vf) else None

                    def meta_str(m: VideoMetadata) -> str:
                        return f"cs={m.colorspace},trc={m.color_trc},pri={m.color_primaries},range={m.color_range}"

                    if meta_str(m1) == meta_str(m2):
                        print(f"  [OK] Metadata Match: {meta_str(m1)}")
                    else:
                        print("  [NG] Metadata Mismatch!")
                        print(f"       File1: {meta_str(m1)}")
                        print(f"       File2: {meta_str(m2)}")

                    if m3 is not None:
                        if meta_str(m1) == meta_str(m3):
                            print(f"  [OK] Metadata Match (CUDA): {meta_str(m1)}")
                        else:
                            print("  [NG] Metadata Mismatch! (CUDA)")
                            print(f"       File1: {meta_str(m1)}")
                            print(f"       File2: {meta_str(m3)}")

                    if meta_str(m1_vf) == meta_str(m2_vf):
                        print(f"  [OK] Metadata Match (vf): {meta_str(m1_vf)}")
                    else:
                        print("  [NG] Metadata Mismatch! (vf)")
                        print(f"       File1: {meta_str(m1_vf)}")
                        print(f"       File2: {meta_str(m2_vf)}")

                    if m3_vf is not None:
                        if meta_str(m1_vf) == meta_str(m3_vf):
                            print(f"  [OK] Metadata Match (CUDA vf): {meta_str(m1_vf)}")
                        else:
                            print("  [NG] Metadata Mismatch! (CUDA vf)")
                            print(f"       File1: {meta_str(m1_vf)}")
                            print(f"       File2: {meta_str(m3_vf)}")

                    # PSNR checks
                    pairs = [
                        (out1, out2, "Quality Pass"),
                        (out1_vf, out2_vf, "Quality Pass (vf)"),
                    ]
                    if os.path.exists(out3):
                        pairs.append((out1, out3, "Quality Pass (CUDA)"))
                    if os.path.exists(out3_vf):
                        pairs.append((out1_vf, out3_vf, "Quality Pass (CUDA vf)"))

                    for p1, p2, label in pairs:
                        res = compare_videos(p1, p2)
                        if res.identical or res.mean_psnr >= 38.0:
                            print(f"  [OK] {label}: PSNR {res.mean_psnr}")
                        else:
                            print(f"  [FAIL] Quality Low ({label}): PSNR {res.mean_psnr}")

    finally:
        orig_req = "requirements.txt"
        if os.path.exists(orig_req):
            run(f"python -m pip install -r {orig_req}")


if __name__ == "__main__":
    main()
