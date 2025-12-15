# Bazzite install (BC-250 / gfx1013)

This flow keeps your nunif checkout and data on the **host** and uses a Distrobox container as an implementation detail. `NUNIF_HOME` controls where models/cache live (defaults to `~/nunif_data`).

## Prereqs (host)
- Bazzite (Fedora-based) with GPU device access (`/dev/kfd`, `/dev/dri`), user in `video`/`render` groups.
- `distrobox` + `podman` available (`sudo dnf install distrobox podman` if needed).
- Clone this repo on the host (e.g. `~/nunif`).

## Install
Run from the repo root on the host:

```bash
bash scripts/install_bazzite.sh
```

What it does:
- Creates/refreshes a Distrobox container (`nunif-bazzite`, image `rockylinux:9` by default) with GPU devices passed through.
- Adds ROCm 6.3 repo + dev toolchain inside the container.
- Creates a host-side venv at `~/.local/share/nunif-bazzite/venv` and installs nunif Python deps.
- Builds **PyTorch and TorchVision for ROCm** with `PYTORCH_ROCM_ARCH=gfx1013` (BC-250) into `~/.cache/nunif-bazzite-build`, then installs those wheels into the venv.
- Installs a host launcher `~/.local/bin/run_iw3_bazzite`.

Knobs:
- `NUNIF_HOME` or `NUNIF_DATA_DIR` – model/data root (default `~/nunif_data`).
- `NUNIF_BAZZITE_CONTAINER`, `NUNIF_BAZZITE_IMAGE`, `NUNIF_BAZZITE_VENV`, `NUNIF_BAZZITE_BUILD` – container/image/venv/build locations.
- `PYTORCH_ROCM_ARCH` (default `gfx1013`), `PYTORCH_BRANCH`, `TORCHVISION_BRANCH`, `MAX_JOBS`.
- `--skip-torch` – skip building/installing torch/torchvision (only if you already installed them in the venv).
- `--torch-wheels DIR` – install torch/torchvision wheels from DIR instead of building.
- `--rebuild-torch` – force rebuild of torch/torchvision even if wheels already exist.

Where things live (host-visible):
- Repo: your checkout (e.g. `~/nunif`).
- Data/models/cache: `NUNIF_HOME` (default `~/nunif_data`).
- Venv: `~/.local/share/nunif-bazzite/venv`.
- Torch build cache/wheels: `~/.cache/nunif-bazzite-build`.
- Host launcher: `~/.local/bin/run_iw3_bazzite`.

If distrobox is missing on an immutable system:
- Install via `sudo rpm-ostree install distrobox podman` and reboot, then rerun the installer.
- (If rpm-ostree is unavailable) install distrobox+podman with your package manager and rerun.

## Run iw3 GUI (host)

```bash
bash scripts/run_iw3_bazzite.sh
# or, after PATH includes ~/.local/bin:
run_iw3_bazzite
```

The launcher:
- Enters the Distrobox container.
- Activates the venv on the host path.
- Sets `NUNIF_HOME` (defaults to `~/nunif_data`).
- Runs `python -m iw3.gui`.

GUI notes:
- Distrobox passes through Wayland/X11 by default; if you use Wayland-only sessions, ensure `XDG_RUNTIME_DIR` and display sockets are shared (Distrobox handles this by default).
- The container is created with GPU device flags so ROCm can see the BC-250.
- File dialogs in iw3 operate on your normal host paths since the repo/data directories are host-mounted.

## gfx1013 ROCm/PyTorch specifics
- The installer builds PyTorch/TorchVision from source with `PYTORCH_ROCM_ARCH=gfx1013` to avoid the missing-kernel “invalid device function” issue seen with upstream ROCm wheels.
- Wheels are cached under `~/.cache/nunif-bazzite-build/pytorch/dist` and `.../vision/dist`. Reuse them by passing `--torch-wheels <dir>` on subsequent installs.
- To rebuild against a different branch or arch, set `PYTORCH_BRANCH`, `TORCHVISION_BRANCH`, `PYTORCH_ROCM_ARCH`, and optionally `--rebuild-torch`.

## Updating / maintenance
- Re-run `bash scripts/install_bazzite.sh --rebuild-torch` after updating PyTorch branches or changing ROCm arch.
- To move data/models, set `NUNIF_HOME=/path/to/data` before running the installer/launcher; the installer will create the directory if missing.
