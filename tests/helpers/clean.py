import argparse
import os
import shutil

# Resolve tests/data directory relative to this file location (tests/helpers/clean.py -> tests/data)
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


def _validate_safe_path(target_path: str, base_dir: str) -> str:
    target_abs = os.path.realpath(os.path.abspath(target_path))
    base_abs = os.path.realpath(os.path.abspath(base_dir))

    # Safety check: base_dir must end with tests/data
    parts = os.path.normpath(base_abs).split(os.sep)
    if len(parts) < 2 or parts[-1] != "data" or parts[-2] != "tests":
        raise ValueError(f"Safety check failed: base_dir must be a 'tests/data' directory, got '{base_abs}'")

    # Safety check: target must be inside or equal to base_dir
    common = os.path.commonpath([target_abs, base_abs])
    if common != base_abs:
        raise ValueError(f"Safety check failed: target '{target_abs}' is outside '{base_abs}'")

    return target_abs


def clean_test_outputs(base_dir: str = DEFAULT_DATA_DIR) -> list[str]:
    base_abs = os.path.realpath(os.path.abspath(base_dir))
    if not os.path.exists(base_abs):
        return []

    _validate_safe_path(base_abs, base_abs)
    removed: list[str] = []

    # Standalone output directories
    output_dir_names = ["race_condition_test", "test_depth_models", "smoke_hwaccel"]
    for name in output_dir_names:
        target = _validate_safe_path(os.path.join(base_abs, name), base_abs)
        if target != base_abs and os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
            removed.append(target)

    # Subdirectories within fixture directories that contain test outputs
    fixture_dirs = ["smoke", "race_condition"]
    for fixture in fixture_dirs:
        parent = os.path.join(base_abs, fixture)
        if os.path.isdir(parent):
            for entry in os.scandir(parent):
                if entry.is_dir():
                    target = _validate_safe_path(entry.path, base_abs)
                    if target != base_abs:
                        shutil.rmtree(target, ignore_errors=True)
                        removed.append(target)

    return removed


def clean_all_test_data(base_dir: str = DEFAULT_DATA_DIR) -> list[str]:
    base_abs = os.path.realpath(os.path.abspath(base_dir))
    if not os.path.exists(base_abs):
        return []

    target = _validate_safe_path(base_abs, base_abs)
    removed: list[str] = []
    shutil.rmtree(target, ignore_errors=True)
    removed.append(target)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean tests/data outputs or all test data")
    parser.add_argument("--all", action="store_true", help="Remove all test data including generated fixtures")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_DATA_DIR, help="Base test data directory")
    args = parser.parse_args()

    if args.all:
        removed = clean_all_test_data(args.base_dir)
    else:
        removed = clean_test_outputs(args.base_dir)

    if removed:
        for p in removed:
            print(f"Removed: {p}")
        print(f"Cleaned {len(removed)} path(s).")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
