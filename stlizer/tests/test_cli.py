import os
from pathlib import Path

from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data


def test_cli(tmp_path: Path) -> None:
    smoke_dir = str(tmp_path / "smoke")
    ensure_smoke_data(smoke_dir)
    test_video = os.path.join(smoke_dir, "sd.mkv")
    out_video = str(tmp_path / "out.mkv")
    run(["python", "-m", "stlizer", "-i", test_video, "-o", out_video])
