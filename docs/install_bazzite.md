# Bazzite (Fedora Atomic) install

This path targets Bazzite on AMD BC-250 (gfx1013) and builds PyTorch with ROCm support inside a Distrobox container so iw3 works from your normal host session.

## Quick start

```bash
git clone https://github.com/stsnewyork83/nunif.git
cd nunif
./scripts/install_bazzite.sh

# run after install
iw3-gui          # GUI
iw3-nunif -h     # CLI help
nunif-torch-check
```

## What the installer sets up

- Creates a Distrobox named `nunif-bazzite` (change with `NUNIF_BAZZITE_CONTAINER`) using `rockylinux:9`.
- Adds AMD’s ROCm 6.3 yum repo inside the container and installs the ROCm toolchain and GUI runtime deps.
- Builds PyTorch/torchvision from source with `PYTORCH_ROCM_ARCH=gfx1013`.
- Creates a shared virtualenv at `~/.local/share/nunif-bazzite/venv` and installs nunif + GUI deps.
- Installs host wrappers in `~/.local/bin`: `iw3-nunif`, `iw3-gui`, `nunif-torch-check`.

## Prerequisites

- Bazzite (Fedora-based) host with BC-250 (gfx1013) and AMD drivers/ROCm stack available on the host.
- `podman` + `distrobox` (installer will try `sudo dnf install -y distrobox podman` if missing).
- `~/.local/bin` on your PATH so the wrappers are discoverable.
- Enough disk/RAM for a PyTorch source build (ROCm build is heavy).

## Customisation knobs

Set any of these env vars before running the installer:

- `NUNIF_BAZZITE_CONTAINER` – Distrobox name (default `nunif-bazzite`).
- `NUNIF_BAZZITE_IMAGE` – Base image (default `rockylinux:9`).
- `NUNIF_BAZZITE_VENV` – Virtualenv path (default `~/.local/share/nunif-bazzite/venv`).
- `NUNIF_BAZZITE_BUILD` – Build cache dir (default `~/.cache/nunif-bazzite-build`).
- `PYTORCH_BRANCH` / `TORCHVISION_BRANCH` – PyTorch/torchvision git tags to build (defaults `v2.4.1` / `v0.19.1`).
- `PYTORCH_ROCM_ARCH` – GPU arch list (default `gfx1013`).
- `MAX_JOBS` – Parallel build jobs (defaults to `nproc` inside the container).
- To mirror `requirements-torch-rocm.txt` pins when they become available, set `PYTORCH_BRANCH=v2.7.1` and `TORCHVISION_BRANCH=v0.22.1` (or your desired releases).

## Running iw3

- CLI: `iw3-nunif -i <input> -o <output>`.
- GUI: `iw3-gui` (Wayland/X11 both work; runs from inside the container but uses your host $HOME).
- Torch sanity: `nunif-torch-check` to confirm ROCm kernels work on gfx1013.
- Optional: set `NUNIF_HOME` if you want models/cache stored away from the repo.

## Troubleshooting

- **HIP invalid device function / missing kernels**: rerun the installer with `PYTORCH_ROCM_ARCH=gfx1013` (default) or clear the venv and rerun so torch rebuilds. Verify with `nunif-torch-check`.
- **Container cannot see the GPU**: ensure `/dev/kfd` and `/dev/dri` exist on the host and your user is in `video`/`render` groups. Recreate the Distrobox so the devices are passed through.
- **wxPython/GUI import errors**: ensure `gstreamer`/`gtk3` runtime packages installed inside the container (installer does this). If DISPLAY/Wayland issues appear, try `GDK_BACKEND=x11 iw3-gui` from the host.
- **No internet in container**: pre-pull the base image (`podman pull rockylinux:9`) and ensure the host can reach `repo.radeon.com` for ROCm packages.
