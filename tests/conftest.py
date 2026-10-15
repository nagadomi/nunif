import os

import pytest

from tests.helpers.clean import DEFAULT_DATA_DIR, clean_all_test_data, clean_test_outputs
from tests.helpers.race_condition_data import ensure_race_condition_data
from tests.helpers.smoke_data import ensure_smoke_data


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--clean",
        action="store_true",
        default=False,
        help="Clean test output directories before running tests",
    )
    parser.addoption(
        "--clean-all",
        action="store_true",
        default=False,
        help="Clean all test data (including generated fixtures in tests/data)",
    )
    parser.addoption(
        "--clean-only",
        action="store_true",
        default=False,
        help="Clean test output directories and exit without running tests",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    clean_all = config.getoption("--clean-all")
    clean = config.getoption("--clean")
    clean_only = config.getoption("--clean-only")

    if clean_all:
        removed = clean_all_test_data()
        if removed:
            print(f"[clean-all] Removed {len(removed)} test data directory.")
    elif clean or clean_only:
        removed = clean_test_outputs()
        if removed:
            print(f"[clean] Removed {len(removed)} output path(s).")

    if clean_only:
        pytest.exit("Test data cleaned.", returncode=0)


@pytest.fixture(scope="session")
def smoke_data_dir() -> str:
    data_dir = os.path.join(DEFAULT_DATA_DIR, "smoke")
    os.makedirs(data_dir, exist_ok=True)
    ensure_smoke_data(base_dir=data_dir)
    return data_dir


@pytest.fixture(scope="session")
def race_condition_data_dir() -> str:
    data_dir = os.path.join(DEFAULT_DATA_DIR, "race_condition")
    os.makedirs(data_dir, exist_ok=True)
    ensure_race_condition_data(base_dir=data_dir)
    return data_dir
