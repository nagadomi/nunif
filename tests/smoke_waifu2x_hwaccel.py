from tests.shell_utils import run

TEST_VIDEO = "tests/data/smoke/sd.mkv"
TEST_DIR = "tests/data/smoke"
OUTPUT_DIR = "tests/data/smoke_hwaccel/waifu2x"
MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"

HWACCEL = "--hwaccel cuda"
H264_ENC = "h264_nvenc"


def main() -> None:
    print(f"**** {__file__}")
    cli = f"python -m waifu2x.cli -y -o {OUTPUT_DIR} --model-dir {MODEL_DIR} --video-codec {H264_ENC} {HWACCEL}"

    run("python -m waifu2x.download_models")
    run(f"{cli} -i {TEST_VIDEO} --noise-level 1 -m noise")
    run(f"{cli} -i {TEST_DIR} -m scale")


if __name__ == "__main__":
    main()
