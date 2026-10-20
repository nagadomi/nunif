from pathlib import Path

from tests.helpers.smoke_data import ensure_smoke_data, verify_all_smoke_data


def test_smoke_data_integrity(tmp_path: Path) -> None:
    smoke_dir = str(tmp_path / "smoke")
    ensure_smoke_data(base_dir=smoke_dir)
    assert verify_all_smoke_data(base_dir=smoke_dir)
