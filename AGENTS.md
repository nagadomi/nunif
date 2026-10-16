# Agent Guidelines

## Python Environment & Execution

- **Python Version**:
  - Python 3.12 is the minimum supported version. Development is primarily done on Python 3.12, and the Windows distribution package bundles Python 3.12. Ensure code is compatible with Python 3.12.
- **Linux / POSIX**:
  - Virtual environments (`venv`) are assumed. Use `.venv` at the project root if it exists (e.g., `.venv/bin/python`).
- **Windows (Distribution Package)**:
  - When working within or maintaining the standalone Windows distribution package (under `windows_package/` or the extracted `nunif-windows` package), follow [`windows_package/AGENTS.md`](./windows_package/AGENTS.md) for environment setup, command execution (`.\nunif-cmd.bat`), testing, and update architecture.
- **Development Dependencies**:
  - Development tools listed in `requirements-dev.txt` (such as `mypy`, `ruff`, `pytest`, and type stubs) are not installed by default in user environments on either Windows or Linux.
  - If tools like `mypy` or `ruff` are required but not installed, install them using `python -m pip install -r requirements-dev.txt` (or install specific packages). For Windows distribution packages, follow [`windows_package/AGENTS.md`](./windows_package/AGENTS.md) to avoid dependency conflicts.
- **Module Execution**:
  - Always run CLI and GUI tools as modules from the project root (e.g., `python -m iw3.cli`, `python -m iw3.gui`, `python -m waifu2x.cli`). Do NOT invoke scripts directly by file path (e.g., `python iw3/cli.py`), which causes relative import errors.

## Forbidden Actions / Directory Restrictions

- **Never access, search, or scan outside the project directory**:
  - All operations (reading, editing, file searching with `find`/`grep`, etc.) must be strictly confined within the project root.
  - Never inspect or traverse outside the project directory, including parent directories (`..`), the user home directory (`~`, `$HOME`), `/tmp`, or other projects.
- **Never list, find, or recursively scan restricted directories**:
  - Do NOT run directory listing (`ls`, `dir`), recursive search (`find`, glob, `grep -r`), or tree exploration on `tmp/`, `data/`, `models/`, or any pretrained model directories (e.g., `*pretrained_models/`).
  - **Reason**: These directories may contain private filenames or massive image datasets (over 10 million files), leading to context overflows, performance degradation, and privacy leaks.
  - Accessing a specific, known file path directly is permitted when required, but exploratory listing or searching within these directories is strictly prohibited.

## Coding Style & Formatting

- **Comments**:
  - Write all code comments in English.
  - Keep comments minimal. Do not explain obvious logic that is clear from reading the code.
  - Do NOT use numbered step comments (e.g., `1. step 1`, `2. step 2`), as they easily become out of sync during modifications.
  - If existing comments become inconsistent with updated code, update or remove them.
- **Lint & Formatting**:
  - The codebase is fully formatted and linted with `ruff`. Ensure changed or created files conform to `ruff check --fix <file>` and `ruff format <file>`.
  - The repository includes `.git-blame-ignore-revs` to ignore bulk formatting commits. Enable it locally with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.
- **Type Hints**:
  - Add type hints to newly created code and newly added/heavily modified functions. Do not retroactively add type hints to untouched existing code to avoid unnecessary diffs.
  - Use modern Python 3.12 syntax, such as PEP 604 union types (`int | None` instead of `Optional[int]`/`Union[int, None]`) and built-in generics (`list[str]`, `dict[str, Any]`).
  - Verify with `mypy <file>` or the full `mypy` command; ensure that no new type errors are introduced (the whole project currently passes `mypy`).

## Core Libraries & Data Types

- **Tensors & Arrays**:
  - Prefer PyTorch tensors (`torch.Tensor`) wherever possible.
  - Use `numpy` only when required for interfacing with third-party libraries.
- **Image Processing**:
  - Use `PIL` (`Pillow`) or `torchvision` for image manipulation.
  - Do NOT use `OpenCV` (`cv2`).
  - `wand` (ImageMagick) is reserved for specific requirements like high-quality downscaling (as in `waifu2x`); do not introduce it for general image processing.
- **Video Processing**:
  - Use `av` (PyAV) and `nunif.utils.video` for video decoding, encoding, and frame manipulation.
  - Do NOT use `OpenCV` (`cv2`).
  - When creating entry points using both PyAV and PyTorch CUDA, call `nunif.utils.video.pyav_init_cuda_primary_context()` before initializing CUDA.
- **GUI**:
  - Use `wxPython` for graphical user interfaces.
