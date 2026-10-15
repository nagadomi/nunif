# Agent Guidelines for Windows Distribution Package

> **Note**: For general coding style, module execution rules, and restricted directories, always follow the primary [`AGENTS.md`](../AGENTS.md) (or [`nunif/AGENTS.md`](./nunif/AGENTS.md)).

## Scope
This document provides specific instructions when maintaining or working inside the **Windows standalone distribution package** (the `windows_package/` folder in the repository, or the root of the extracted `nunif-windows` package).

## Context-Specific Instructions

### 1. When workspace root is the distribution package (`nunif-windows`)
- **Git & Source Root**:
  - The actual Git repository and source code are located in the `nunif/` subdirectory.
  - Run Git commands targeting `nunif/`. You can execute `.\nunif-cmd.bat git <args>` from the package root to automatically use the bundled MinGit inside `nunif/`.
- **Upstream Source of Root Files**:
  - Root batch files (`*.bat`), `docs/`, and `torch_compile/` are deployed from `nunif/windows_package/`.
  - When modifying these files, always make changes to `nunif/windows_package/` so they can be committed to the Git repository.
  - To test changes locally in the distribution root, copy the modified files from `nunif/windows_package/` to the root (or run `update-installer.bat`).
- **Command Execution**:
  - Always use `.\nunif-cmd.bat <command>` (e.g., `.\nunif-cmd.bat python -m iw3.cli ...`) to run commands in the bundled Python environment.
  - `nunif-cmd.bat` is located in the package root and is NOT in `PATH`. Always invoke it with `.\` (e.g., `.\nunif-cmd.bat`) from the root directory, especially in PowerShell where current directory binaries require explicit paths.
  - `.\nunif-cmd.bat` automatically sets environment variables and switches the working directory to `nunif/`.
  - Do NOT execute `nunif-prompt.bat` (it starts an interactive shell and blocks agent execution).
  - Do NOT run `update.bat` arbitrarily during development, as it will run `git reset --hard` and discard uncommitted changes.
- **Development & Testing Dependencies**:
  - Development and testing tools (`pytest`, `ruff`, `mypy`, and type stubs in `requirements-dev.txt`) are NOT installed by default in the distribution package.
  - Install specific tools on demand:
    - `.\nunif-cmd.bat python -m pip install pytest ruff mypy types-pywin32`
  - **Caution**: Prefer installing specific tools rather than `pip install -r requirements-dev.txt`. Installing unnecessary dev packages may introduce dependency conflicts with bundled packages (such as bundled `opencv-python` requiring `numpy>=2`).
- **Running Tests**:
  - Run tests with `.\nunif-cmd.bat python -m pytest <test_path>` (e.g., `.\nunif-cmd.bat python -m pytest tests/integration/waifu2x/test_cli.py`).
  - Tests work in CPU-only environments without a GPU (execution will take longer).
  - Clean test output directories using `.\nunif-cmd.bat python -m pytest --clean-only`.
- **Update Mechanism Architecture**:
  - Batch files cannot safely overwrite themselves while running on Windows (causes corrupted execution due to cmd.exe offset reading).
  - Therefore, the update process is decoupled: `update.bat` updates the code and copies `update-installer.bat`, while `update-installer.bat` updates `update.bat` and other launcher batch files. Do not merge them into a single self-modifying batch script.
- **Directory Restrictions Reminder**:
  - In addition to the restrictions in `nunif/AGENTS.md`, never recursively scan or list `tmp/`, `python/`, or `git/` in the package root.

### 2. When workspace root is the main repository (`nunif`)
- The `windows_package/` directory contains templates, packaging scripts, and installers for Windows distribution releases.
- Editing files under `windows_package/` modifies the source templates for the Windows package.
- All core development guidelines remain defined in the project root [`AGENTS.md`](../AGENTS.md).
