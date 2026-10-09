# Agent Guidelines

## Python Environment & Execution

- **Python Version**:
  - Python 3.12 is the minimum supported version. Development is primarily done on Python 3.12, and the Windows distribution package bundles Python 3.12. Ensure code is compatible with Python 3.12.
- **Linux / POSIX**:
  - Virtual environments (`venv`) are assumed. Use `.venv` at the project root if it exists (e.g., `.venv/bin/python`).
- **Windows (Distribution Package)**:
  - Users typically run within the standalone distribution environment under `windows_package/`, which contains bundled embedded Python and isolates environment variables via `setenv.bat`.
  - To run non-interactive commands (such as `python -c "..."` or scripts), use `nunif-cmd.bat <command>`.
  - Do NOT call `nunif-prompt.bat`, as it starts an interactive `cmd` shell and will block execution.
- **Development Dependencies**:
  - Development tools listed in `requirements-dev.txt` (such as `mypy`, `ruff`, and type stubs) are not installed by default in user environments on either Windows or Linux.
  - If tools like `mypy` or `ruff` are required but not installed, install them using `python -m pip install -r requirements-dev.txt` (or install the specific package needed). On Windows distribution packages, execute this via `nunif-cmd.bat python -m pip install ...` (prefer `python -m pip` over `pip`).
- **Module Execution**:
  - Always run CLI and GUI tools as modules from the project root (e.g., `python -m iw3.cli`, `python -m iw3.gui`, `python -m waifu2x.cli`). Do NOT invoke scripts directly by file path (e.g., `python iw3/cli.py`), which causes relative import errors.

## Forbidden Actions / Directory Restrictions

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
  - Run `ruff check --fix <file>` and `ruff format <file>` on changed or created files only.
  - Do NOT run `ruff` across the entire repository or untouched files, as existing code may not be formatted and will produce large unintended diffs.
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
