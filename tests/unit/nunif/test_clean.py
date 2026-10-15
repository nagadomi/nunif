import os

import pytest

from tests.helpers.clean import (
    DEFAULT_DATA_DIR,
    _validate_safe_path,
    clean_all_test_data,
    clean_test_outputs,
)


def test_default_data_dir() -> None:
    assert os.path.isabs(DEFAULT_DATA_DIR)
    assert DEFAULT_DATA_DIR.endswith(os.path.join("tests", "data"))


def test_validate_safe_path_rejects_unsafe_paths() -> None:
    # Cannot clean root or system directories
    with pytest.raises(ValueError, match="Safety check failed"):
        _validate_safe_path("/", "/")

    # Cannot clean arbitrary directories outside tests/data
    with pytest.raises(ValueError, match="Safety check failed"):
        _validate_safe_path("/tmp", "/tmp")

    # Cannot clean outside base_dir even if base_dir is tests/data
    with pytest.raises(ValueError, match="Safety check failed"):
        _validate_safe_path("/tmp/malicious", DEFAULT_DATA_DIR)

    # Valid target within tests/data passes
    valid_sub = os.path.join(DEFAULT_DATA_DIR, "smoke", "iw3")
    assert _validate_safe_path(valid_sub, DEFAULT_DATA_DIR) == os.path.realpath(valid_sub)


def test_clean_functions_reject_unsafe_base_dir() -> None:
    with pytest.raises(ValueError, match="Safety check failed"):
        clean_test_outputs(base_dir="/tmp")

    with pytest.raises(ValueError, match="Safety check failed"):
        clean_all_test_data(base_dir="/tmp")
