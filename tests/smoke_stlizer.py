import os

from tests.shell_utils import run

TEST_VIDEO = "tests/data/smoke/sd.mkv"
OUTPUT_DIR = "tests/data/smoke/stlizer"


def main() -> None:
    print(f"**** {__file__}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run(f"python -m stlizer -i {TEST_VIDEO} -o {os.path.join(OUTPUT_DIR, 'out.mkv')}")


if __name__ == "__main__":
    main()
