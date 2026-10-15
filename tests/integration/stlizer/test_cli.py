import os

from tests.helpers.runner import run

OUTPUT_DIR = "tests/data/smoke/stlizer"


def test_cli(smoke_data_dir: str) -> None:
    test_video = os.path.join(smoke_data_dir, "sd.mkv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_video = os.path.join(OUTPUT_DIR, "out.mkv")
    run(f"python -m stlizer -i {test_video} -o {out_video}")
