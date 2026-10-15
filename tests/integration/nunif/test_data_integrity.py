from tests.helpers.smoke_data import verify_all_smoke_data


def test_smoke_data_integrity(smoke_data_dir: str) -> None:
    assert verify_all_smoke_data(base_dir=smoke_data_dir)
