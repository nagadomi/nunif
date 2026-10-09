import os

from tests.shell_utils import run

TEST_VIDEO = "tests/data/smoke/sd.mkv"
TEST_VIDEO_HDR = "tests/data/smoke/hdr.mkv"
TEST_DIR = "tests/data/smoke"
OUTPUT_DIR = "tests/data/smoke_hwaccel/iw3"
TEST_YAML = os.path.join(OUTPUT_DIR, "sd", "iw3_export.yml")

H264_ENC = "h264_nvenc"
H265_ENC = "hevc_nvenc"
HWACCEL = "--hwaccel cuda"


def main() -> None:
    print(f"**** {__file__}")
    cli = f"python -m iw3.cli -y -o {OUTPUT_DIR}"
    h264 = f"--video-codec {H264_ENC} {HWACCEL}"
    h265 = f"--video-codec {H265_ENC} {HWACCEL}"

    # base
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata {h264}")
    run(f"python -m iw3 -y -i {TEST_DIR} -o {OUTPUT_DIR} --depth-model Any_S --metadata {h265}")

    # EMA
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10 {h264}")

    # Batch
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream {h264}")

    # Low VRAM
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --low-vram {h264}")

    # VDA
    run(
        f"{cli} -i {TEST_VIDEO} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10"
        f" --scene-detect --disable-scene-cache {h264}"
    )

    # VDA Stream
    run(
        f"{cli} -i {TEST_VIDEO} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10"
        f" --scene-detect {h264}"
    )

    # HDR
    run(f"{cli} -i {TEST_VIDEO_HDR} --depth-model Any_S --colorspace auto --pix-fmt yuv420p10le {h265}")
    run(
        f"{cli} -i {TEST_VIDEO_HDR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10"
        f" --scene-detect --pix-fmt yuv420p10le {h265}"
    )

    # HDR2SDR
    run(f"{cli} -i {TEST_VIDEO_HDR} --depth-model Any_S --colorspace bt709-tv --pix-fmt yuv420p {h265}")

    # inpaint
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --method forward_inpaint {h264}")
    run(
        f"{cli} -i {TEST_VIDEO} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10"
        f" --scene-detect {h265}"
    )

    # export
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --export-disparity {HWACCEL}")
    run(
        f"{cli} -i {TEST_VIDEO_HDR} --depth-model VDA_S --export --export-depth-only --export-depth-fit"
        f" --ema-normalize --ema-buffer 10 --scene-detect {HWACCEL}"
    )

    # import
    run(f"{cli} -i {TEST_YAML} --video-codec {H264_ENC}")

    # vf
    run(f'{cli} -i {TEST_VIDEO} --depth-model Any_S --vf "scale=-2:320,crop=256:256" {HWACCEL}')

    # keyframe
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --keyframe {HWACCEL}")


if __name__ == "__main__":
    main()
