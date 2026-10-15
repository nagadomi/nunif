# Tests

## Directory Structure

- `tests/unit/`: Fast unit tests (no video processing or CLI subprocesses).
- `tests/smoke/`: End-to-end and integration smoke tests. Subdivided by package (e.g., `iw3/`, `waifu2x/`).
- `tests/tools/`: Regression comparison scripts (e.g. comparing output across git revisions).
- `tests/helpers/`: Shared test utilities, runners, and data generators.

## Running Tests

Run commands from the repository root:

```bash
# Run unit tests only (fast)
pytest tests/unit

# Run smoke tests
pytest tests/smoke
pytest tests/smoke/iw3

# Run all tests
pytest

# Filter by markers (defined in pytest.ini: slow, hwaccel)
pytest -m "not slow"
pytest -m "hwaccel"
```

## Comparison Tools

Regression tools under `tests/tools/` compare outputs against a baseline git revision (default: `9fd1812477a2a7f372c87ac7ac14fdfab0a7c3ed`) via `git worktree`:

```bash
# Compare current working directory against default baseline commit
python -m tests.tools.compare_iw3_pipeline path/to/video.mp4

# Specify a custom baseline or target revision/directory
python -m tests.tools.compare_iw3_pipeline path/to/video.mp4 --ref HEAD~1
python -m tests.tools.compare_iw3_pipeline path/to/video.mp4 --target path/to/dev --ref master
```

