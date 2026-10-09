import os

from tests.shell_utils import run

TEST_IMAGE = "tests/data/smoke/sd.png"
TEST_VIDEO = "tests/data/smoke/sd.mkv"
TEST_VIDEO_HDR = "tests/data/smoke/hdr.mkv"
TEST_DIR = "tests/data/smoke"
OUTPUT_DIR = "tests/data/smoke/iw3"


def main() -> None:
    print(f"**** {__file__}")
    cli = f"python -m iw3.cli -y -o {OUTPUT_DIR}"

    # base
    run(f"{cli} -i {TEST_IMAGE} --depth-model Any_S --metadata")
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata")
    run(f"python -m iw3 -y -i {TEST_DIR} -o {OUTPUT_DIR} --depth-model Any_S --metadata")

    # EMA
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10")

    # Batch
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream")

    # Low VRAM
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --metadata --low-vram")

    # VDA
    run(
        f"{cli} -i {TEST_VIDEO} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10"
        " --scene-detect --disable-scene-cache"
    )

    # VDA Stream
    run(f"{cli} -i {TEST_VIDEO} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect")

    # HDR
    run(f"{cli} -i {TEST_VIDEO_HDR} --depth-model Any_S --colorspace auto --video-codec libx265 --pix-fmt yuv420p10le")
    run(
        f"{cli} -i {TEST_VIDEO_HDR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10"
        " --scene-detect --video-codec libx265 --pix-fmt yuv420p10le"
    )

    # HDR2SDR
    run(f"{cli} -i {TEST_VIDEO_HDR} --depth-model Any_S --colorspace bt709-tv --video-codec libx265 --pix-fmt yuv420p")

    # inpaint
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --method forward_inpaint")
    run(
        f"{cli} -i {TEST_VIDEO} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10"
        " --scene-detect"
    )

    # export
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --export-disparity --export-depth-only --export-depth-fit")
    run(f"{cli} -i {TEST_VIDEO_HDR} --depth-model VDA_S --export --ema-normalize --ema-buffer 10 --scene-detect")
    run(f"{cli} -i {TEST_IMAGE} --depth-model Any_S --export --depth-aa --export-depth-fit")

    # import
    run(f"{cli} -i {os.path.join(OUTPUT_DIR, 'hdr', 'iw3_export.yml')}")
    run(f"{cli} -i {os.path.join(OUTPUT_DIR, 'iw3_export.yml')}")

    # vf
    run(f'{cli} -i {TEST_VIDEO} --depth-model Any_S --vf "scale=-2:320,crop=256:256"')

    # keyframe
    run(f"{cli} -i {TEST_VIDEO} --depth-model Any_S --keyframe")


if __name__ == "__main__":
    main()
