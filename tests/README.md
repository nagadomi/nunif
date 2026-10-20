# Tests

## Directory Structure

- `tests/images/`: Shared test fixture images.
- `tests/helpers/`: Shared test utilities, runners, and data generators.
- Package test suites:
  - `iw3/tests/`: Tests and regression tools (`iw3/tests/tools/`) for iw3.
  - `iw3/player/tests/`: Tests for iw3-player WebXR server and APIs.
  - `waifu2x/tests/`: Tests for waifu2x.
  - `stlizer/tests/`: Tests for stlizer.
  - `nunif/tests/`: Tests for nunif core framework.

## Running Tests

Run commands from the repository root:

```bash
# Run specific package tests
pytest iw3/tests
pytest iw3/player/tests
pytest waifu2x/tests
pytest nunif/tests
pytest stlizer/tests

# Run multiple package tests
pytest iw3/tests waifu2x/tests

# Run all tests
pytest

# Filter by markers (defined in pytest.ini: slow, hwaccel)
pytest iw3/tests -m "not slow"
pytest waifu2x/tests -m "hwaccel"
```

> **Note**: Test inputs (smoke fixtures) and outputs are automatically created in `pytest`'s temporary directory (`tmp_path`) and cleaned up after test runs. No manual cleanup is required.

## Comparison Tools

Regression tools under `iw3/tests/tools/` compare outputs against a baseline git revision (default: `9fd1812477a2a7f372c87ac7ac14fdfab0a7c3ed`) via `git worktree`:

```bash
# Compare current working directory against default baseline commit
python -m iw3.tests.tools.compare_pipeline path/to/video.mp4

# Specify a custom baseline or target revision/directory
python -m iw3.tests.tools.compare_pipeline path/to/video.mp4 --ref HEAD~1
python -m iw3.tests.tools.compare_pipeline path/to/video.mp4 --target path/to/dev --ref master
```
