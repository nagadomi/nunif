import os

import pytest

from tests.helpers.race_condition_data import ensure_race_condition_data
from tests.helpers.smoke_data import ensure_smoke_data


@pytest.fixture(scope="session")
def smoke_data_dir() -> str:
    data_dir = "tests/data/smoke"
    os.makedirs(data_dir, exist_ok=True)
    ensure_smoke_data(base_dir=data_dir)
    return data_dir


@pytest.fixture(scope="session")
def race_condition_data_dir() -> str:
    data_dir = "tests/data/race_condition"
    os.makedirs(data_dir, exist_ok=True)
    ensure_race_condition_data(base_dir=data_dir)
    return data_dir
