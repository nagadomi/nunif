#!/usr/bin/env bash
set -euo pipefail
# Bazzite/Fedora installer for nunif with ROCm (gfx1013/BC-250) support.
# Run this on the host. It will create/refresh a Distrobox container, set up a
# virtualenv on the host filesystem, and optionally build PyTorch+TorchVision
# for gfx1013. The iw3 GUI will run against host-visible paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BOX_NAME="${NUNIF_BAZZITE_CONTAINER:-nunif-bazzite}"
BOX_IMAGE="${NUNIF_BAZZITE_IMAGE:-rockylinux:9}"
VENV_DIR="${NUNIF_BAZZITE_VENV:-$HOME/.local/share/nunif-bazzite/venv}"
BUILD_DIR="${NUNIF_BAZZITE_BUILD:-$HOME/.cache/nunif-bazzite-build}"
DATA_DIR_DEFAULT="${NUNIF_DATA_DIR:-${NUNIF_HOME:-$HOME/nunif_data}}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-v2.4.1}"
TORCHVISION_BRANCH="${TORCHVISION_BRANCH:-v0.19.1}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1013}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
TORCH_WHEELS_DIR="${TORCH_WHEELS_DIR:-}"
BUILD_TORCH=1
FORCE_REBUILD=0

usage() {
  cat <<'EOF'
Usage: scripts/install_bazzite.sh [options]

Options:
  --skip-torch         Skip building/installing PyTorch/TorchVision (you can install wheels manually).
  --torch-wheels DIR   Use existing torch/torchvision wheels from DIR instead of building.
  --rebuild-torch      Force a rebuild of torch/torchvision even if wheels exist.
  --container NAME     Override distrobox container name (default: nunif-bazzite).
  --image IMAGE        Override distrobox base image (default: rockylinux:9).
  -h, --help           Show this help.

Environment knobs:
  NUNIF_HOME / NUNIF_DATA_DIR    Host directory for models/cache (default: $HOME/nunif_data).
  NUNIF_BAZZITE_VENV             Virtualenv path (host) (default: ~/.local/share/nunif-bazzite/venv).
  NUNIF_BAZZITE_BUILD            Build cache path (host) (default: ~/.cache/nunif-bazzite-build).
  PYTORCH_ROCM_ARCH              gfx arch for ROCm build (default: gfx1013).
  MAX_JOBS                       Parallel jobs for builds (default: nproc).
EOF
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

ensure_path() {
  case ":${PATH}:" in
    *:"$HOME/.local/bin":*) ;;
    *) export PATH="$HOME/.local/bin:${PATH}" ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-torch) BUILD_TORCH=0; shift ;;
      --torch-wheels) TORCH_WHEELS_DIR="$2"; BUILD_TORCH=0; shift 2 ;;
      --rebuild-torch) FORCE_REBUILD=1; shift ;;
      --container) BOX_NAME="$2"; shift 2 ;;
      --image) BOX_IMAGE="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $1" ;;
    esac
  done
}

require_fedora_like() {
  if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" != "fedora" && "${ID:-}" != "bazzite" && "${ID_LIKE:-}" != *fedora* ]]; then
      log "Warning: expected Fedora/Bazzite, detected ${ID:-unknown}. Proceeding anyway."
    fi
  fi
}

ensure_distrobox() {
  if command -v distrobox >/dev/null 2>&1; then
    return
  fi

  # On immutable Fedora/Bazzite, prefer rpm-ostree so binaries land in /usr.
  if command -v rpm-ostree >/dev/null 2>&1; then
    cat <<'EOF' >&2
distrobox not found. On Bazzite/Silverblue-style systems, install it via:
  sudo rpm-ostree install distrobox podman
Then reboot, ensure distrobox is on PATH, and re-run this installer.
EOF
    exit 1
  fi

  if command -v dnf >/dev/null 2>&1; then
    log "Installing distrobox and podman (sudo required)..."
    sudo dnf install -y distrobox podman || die "Install distrobox/podman manually and re-run."
    return
  fi

  die "distrobox not found and package manager not detected. Install distrobox+podman and re-run."
}

create_container() {
  if distrobox list | grep -E "^[[:space:]]*${BOX_NAME}[[:space:]]" >/dev/null 2>&1; then
    log "Distrobox ${BOX_NAME} already exists."
    return
  fi
  log "Creating distrobox ${BOX_NAME} from ${BOX_IMAGE} ..."
  distrobox create --name "${BOX_NAME}" --image "${BOX_IMAGE}" --init --additional-packages systemd \
    --additional-flags "--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render"
  distrobox enter "${BOX_NAME}" -- /bin/true
}

provision_root() {
  log "Provisioning base packages and ROCm toolchain inside ${BOX_NAME} ..."
  distrobox enter "${BOX_NAME}" -- /bin/bash -lc "
set -euo pipefail
sudo -n true 2>/dev/null || true
sudo dnf -y install epel-release dnf-plugins-core
sudo dnf config-manager --set-enabled crb || true

sudo tee /etc/yum.repos.d/rocm.repo >/dev/null <<'EOF'
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/yum/6.3/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo dnf -y install git gcc gcc-c++ clang make cmake ninja-build patch \
  python3 python3-devel python3-pip python3-virtualenv python3-wheel python3-setuptools \
  rocm-hip-sdk rocm-hip-libraries rocm-hip-libraries-devel rocm-llvm rocblas hipblas hipfft hipsparse hiprand miopen-hip roctracer roctracer-dev \
  ffmpeg SDL2 mesa-libGL mesa-libEGL mesa-libGLU-devel libX11-devel libXcursor-devel libXi-devel libXtst-devel \
  which unzip tar
"
}

ensure_dirs() {
  mkdir -p "${DATA_DIR_DEFAULT}" "${VENV_DIR}" "${BUILD_DIR}"
}

container_exec() {
  distrobox enter "${BOX_NAME}" -- /bin/bash -lc "$*"
}

setup_python_base() {
  log "Setting up virtualenv at ${VENV_DIR} ..."
  container_exec "
set -euo pipefail
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r '${REPO_ROOT}/requirements.txt'
python -m pip install -r '${REPO_ROOT}/requirements-gui.txt'
"
}

maybe_clone_repo() {
  local repo_url="$1" target_dir="$2" branch="$3"
  if [[ -d "${target_dir}/.git" ]]; then
    log "Updating ${target_dir} ..."
    container_exec "cd '${target_dir}' && git fetch --all && git checkout '${branch}' && git pull --ff-only origin '${branch}'"
  else
    log "Cloning ${repo_url} into ${target_dir} ..."
    container_exec "git clone --depth 1 --branch '${branch}' '${repo_url}' '${target_dir}'"
  fi
}

build_torch() {
  local torch_src="${BUILD_DIR}/pytorch"
  if [[ ${FORCE_REBUILD} -eq 0 && -n "$(find "${BUILD_DIR}/pytorch/dist" -maxdepth 1 -name 'torch-*rocm*.whl' -print -quit 2>/dev/null)" ]]; then
    log "Found existing torch wheel, skipping rebuild (use --rebuild-torch to force)."
  else
    maybe_clone_repo https://github.com/pytorch/pytorch.git "${torch_src}" "${PYTORCH_BRANCH}"
    log "Building torch for ROCm (${PYTORCH_ROCM_ARCH}) ..."
    container_exec "
set -euo pipefail
export PATH=/opt/rocm/bin:\$PATH
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH='${PYTORCH_ROCM_ARCH}'
export USE_ROCM=1
export USE_CUDA=0
export BUILD_TEST=0
export USE_MKLDNN=0
export MAX_JOBS='${MAX_JOBS}'
cd '${torch_src}'
git submodule sync --recursive
git submodule update --init --recursive
source '${VENV_DIR}/bin/activate'
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup.py bdist_wheel
"
  fi
}

build_torchvision() {
  local vision_src="${BUILD_DIR}/vision"
  if [[ ${FORCE_REBUILD} -eq 0 && -n "$(find "${BUILD_DIR}/vision/dist" -maxdepth 1 -name 'torchvision-*rocm*.whl' -print -quit 2>/dev/null)" ]]; then
    log "Found existing torchvision wheel, skipping rebuild (use --rebuild-torch to force)."
  else
    maybe_clone_repo https://github.com/pytorch/vision.git "${vision_src}" "${TORCHVISION_BRANCH}"
    log "Building torchvision for ROCm (${PYTORCH_ROCM_ARCH}) ..."
    container_exec "
set -euo pipefail
export PATH=/opt/rocm/bin:\$PATH
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH='${PYTORCH_ROCM_ARCH}'
export USE_ROCM=1
export USE_CUDA=0
export BUILD_TEST=0
export MAX_JOBS='${MAX_JOBS}'
cd '${vision_src}'
git submodule sync --recursive
git submodule update --init --recursive
source '${VENV_DIR}/bin/activate'
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup.py bdist_wheel
"
  fi
}

pick_wheel() {
  local dir="$1" prefix="$2"
  find "${dir}" -maxdepth 1 -type f -name "${prefix}-*rocm*.whl" | sort | tail -n1
}

torch_installed() {
  container_exec "
set -euo pipefail
source '${VENV_DIR}/bin/activate'
python - <<'PY'
import sys
try:
    import torch  # noqa: F401
except Exception:
    sys.exit(1)
PY
" >/dev/null 2>&1
}

install_torch_wheels() {
  local torch_wheel vision_wheel wheel_source
  if [[ -n "${TORCH_WHEELS_DIR}" ]]; then
    wheel_source="${TORCH_WHEELS_DIR}"
  else
    wheel_source="${BUILD_DIR}"
  fi

  torch_wheel="$(pick_wheel "${wheel_source}/pytorch/dist" "torch")"
  vision_wheel="$(pick_wheel "${wheel_source}/vision/dist" "torchvision")"

  [[ -n "${torch_wheel}" ]] || die "torch wheel not found in ${wheel_source}. Build or provide wheels."
  [[ -n "${vision_wheel}" ]] || die "torchvision wheel not found in ${wheel_source}. Build or provide wheels."

  log "Installing torch wheel: ${torch_wheel}"
  log "Installing torchvision wheel: ${vision_wheel}"

  container_exec "
set -euo pipefail
source '${VENV_DIR}/bin/activate'
python -m pip install '${torch_wheel}'
python -m pip install '${vision_wheel}'
"
}

install_wrappers() {
  mkdir -p "$HOME/.local/bin"
  cat >"$HOME/.local/bin/run_iw3_bazzite" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${SCRIPT_DIR}"
REPO_ROOT="${REPO_ROOT}"
BOX_NAME="${BOX_NAME}"
VENV_DIR="${VENV_DIR}"
DATA_DIR="${DATA_DIR_DEFAULT}"
export NUNIF_HOME="\${NUNIF_HOME:-\${NUNIF_DATA_DIR:-\${DATA_DIR}}}"
exec distrobox enter "\${BOX_NAME}" -- /bin/bash -lc "cd '\${REPO_ROOT}' && source '\${VENV_DIR}/bin/activate' && NUNIF_HOME='\${NUNIF_HOME}' python -m iw3.gui \"\$@\""
EOF
  chmod +x "$HOME/.local/bin/run_iw3_bazzite"
  log "Host launcher installed at $HOME/.local/bin/run_iw3_bazzite"
}

main() {
  parse_args "$@"
  ensure_path
  require_fedora_like
  ensure_dirs
  ensure_distrobox
  create_container
  provision_root
  setup_python_base
  if [[ ${BUILD_TORCH} -eq 1 ]]; then
    build_torch
    build_torchvision
    install_torch_wheels
  else
    if [[ -n "${TORCH_WHEELS_DIR}" ]]; then
      install_torch_wheels
    elif torch_installed; then
      log "torch is already installed in the venv; skipping wheel install."
    else
      die "Torch not installed and --skip-torch was requested. Provide wheels via --torch-wheels DIR or allow a build."
    fi
  fi
  install_wrappers
  log "Done. Set NUNIF_HOME to control model/data location (current default: ${DATA_DIR_DEFAULT})."
  log "Launch GUI from host: run_iw3_bazzite"
}

main "$@"
