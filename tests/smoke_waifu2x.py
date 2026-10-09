from tests.shell_utils import run

TEST_IMAGE = "tests/data/smoke/sd.png"
TEST_VIDEO = "tests/data/smoke/sd.mkv"
TEST_DIR = "tests/data/smoke"
OUTPUT_DIR = "tests/data/smoke/waifu2x"
MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"


def main() -> None:
    print(f"**** {__file__}")
    cli = f"python -m waifu2x.cli -y -o {OUTPUT_DIR} --model-dir {MODEL_DIR}"

    run("python -m waifu2x.download_models")
    run(f"{cli} -i {TEST_IMAGE} --noise-level 0 -m noise_scale")
    run(f"{cli} -i {TEST_VIDEO} --noise-level 1 -m noise")
    run(f"{cli} -i {TEST_DIR} -m scale")


if __name__ == "__main__":
    main()
