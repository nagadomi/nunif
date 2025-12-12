#!/usr/bin/env bash
set -euo pipefail

# Bazzite/Fedora installer for nunif with ROCm (gfx1013/BC-250) support.
# This script is intended to be executed on the host (Bazzite) and will
# prepare a Distrobox container with the ROCm toolchain, build PyTorch
# for gfx1013, and drop wrapper scripts into ~/.local/bin to launch iw3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BOX_NAME="${NUNIF_BAZZITE_CONTAINER:-nunif-bazzite}"
BOX_IMAGE="${NUNIF_BAZZITE_IMAGE:-rockylinux:9}"
VENV_DIR="${NUNIF_BAZZITE_VENV:-$HOME/.local/share/nunif-bazzite/venv}"
BUILD_DIR="${NUNIF_BAZZITE_BUILD:-$HOME/.cache/nunif-bazzite-build}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-v2.4.1}"
TORCHVISION_BRANCH="${TORCHVISION_BRANCH:-v0.19.1}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1013}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

ensure_path() {
  case ":${PATH}:" in
    *:"$HOME/.local/bin":*)
      ;;
    *)
      export PATH="$HOME/.local/bin:${PATH}"
      ;;
  esac
}

require_fedora_like() {
  if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" != "fedora" && "${ID:-}" != "bazzite" && "${ID_LIKE:-}" != *fedora* ]]; then
      echo "Warning: This installer targets Bazzite/Fedora. Detected ${ID:-unknown}." >&2
    fi
  fi
}

ensure_distrobox() {
  if ! command -v distrobox >/dev/null 2>&1; then
    echo "Installing distrobox + podman via dnf (requires sudo)..."
    sudo dnf install -y distrobox podman || {
      echo "Failed to install distrobox. Please install distrobox/podman manually and re-run." >&2
      exit 1
    }
  fi
}

create_container() {
  if distrobox list | grep -E "^[[:space:]]*${BOX_NAME}[[:space:]]" >/dev/null 2>&1; then
    echo "Distrobox ${BOX_NAME} already exists."
    return
  fi

  echo "Creating distrobox ${BOX_NAME} from ${BOX_IMAGE} ..."
  distrobox create \
    --name "${BOX_NAME}" \
    --image "${BOX_IMAGE}" \
    --init \
    --additional-flags "--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render"
  distrobox enter "${BOX_NAME}" -- /bin/true
}

provision_root() {
  echo "Provisioning base packages and ROCm toolchain inside ${BOX_NAME} (root)..."
  distrobox enter "${BOX_NAME}" --root -- /bin/bash -lc "
set -euo pipefail
dnf -y install epel-release dnf-plugins-core
dnf config-manager --set-enabled crb || true

cat >/etc/yum.repos.d/rocm.repo <<'EOF'
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/yum/6.3/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

dnf -y install git python3 python3-devel python3-virtualenv \
  gcc gcc-c++ make cmake ninja-build \
  openssl-devel libffi-devel \
  wget curl pciutils numactl-libs

dnf -y install rocm-dev rocm-hip-sdk rocblas-devel hipblas-devel miopen-hip \
  rccl-devel hiprand-devel hipsparse-devel rocsolver-devel rocfft-devel

dnf -y install gtk3 gtk3-devel gstreamer1 gstreamer1-plugins-base \
  pulseaudio-libs alsa-lib \
  libX11-devel libXrandr-devel libXxf86vm-devel libSM-devel libICE-devel \
  libXScrnSaver \
  mesa-libGL-devel mesa-libEGL-devel mesa-libGLU-devel
"
}

provision_user() {
  echo "Provisioning Python environment and building PyTorch inside ${BOX_NAME} ..."
  distrobox enter "${BOX_NAME}" -- /bin/bash -lc "
set -euo pipefail
REPO_ROOT='${REPO_ROOT}'
VENV_DIR='${VENV_DIR}'
BUILD_DIR='${BUILD_DIR}'
PYTORCH_BRANCH='${PYTORCH_BRANCH}'
TORCHVISION_BRANCH='${TORCHVISION_BRANCH}'
PYTORCH_ROCM_ARCH='${PYTORCH_ROCM_ARCH}'
MAX_JOBS='${MAX_JOBS}'

mkdir -p \"\${BUILD_DIR}\" \"\$(dirname \"\${VENV_DIR}\")\"

python3 -m venv \"\${VENV_DIR}\"
source \"\${VENV_DIR}/bin/activate\"
python -m pip install --upgrade pip setuptools wheel ninja

echo 'Checking existing torch installation...'
if ! python - <<'PY'
import sys
try:
    import torch
    import torch.cuda
    torch.zeros(1, device='cuda')
    print('Existing torch found:', torch.__version__)
    sys.exit(0)
except Exception as e:
    print('Torch check failed, will (re)build:', e)
    sys.exit(1)
PY
then
  if [ ! -d \"\${BUILD_DIR}/pytorch\" ]; then
    git clone --recursive --branch \"\${PYTORCH_BRANCH}\" https://github.com/pytorch/pytorch.git \"\${BUILD_DIR}/pytorch\"
  else
    cd \"\${BUILD_DIR}/pytorch\"
    git fetch --depth 1 origin \"\${PYTORCH_BRANCH}\"
    git checkout \"\${PYTORCH_BRANCH}\"
    git submodule sync --recursive
    git submodule update --init --recursive
  fi
  cd \"\${BUILD_DIR}/pytorch\"
  python -m pip install -r requirements.txt
  export USE_ROCM=1
  export USE_CUDA=0
  export BUILD_TEST=0
  export BUILD_CAFFE2=0
  export PYTORCH_ROCM_ARCH
  export ROCM_PATH=/opt/rocm
  export CMAKE_PREFIX_PATH=\$(python -c 'import sys; print(sys.prefix)')
  export MAX_JOBS
  python setup.py develop
fi

echo 'Checking existing torchvision installation...'
if ! python - <<'PY'
import sys
try:
    import torchvision
    import torch
    _ = torchvision.ops.nms
    print('Existing torchvision found:', torchvision.__version__)
    sys.exit(0)
except Exception as e:
    print('torchvision check failed, will (re)build:', e)
    sys.exit(1)
PY
then
  if [ ! -d \"\${BUILD_DIR}/vision\" ]; then
    git clone --recursive --branch \"\${TORCHVISION_BRANCH}\" https://github.com/pytorch/vision.git \"\${BUILD_DIR}/vision\"
  else
    cd \"\${BUILD_DIR}/vision\"
    git fetch --depth 1 origin \"\${TORCHVISION_BRANCH}\"
    git checkout \"\${TORCHVISION_BRANCH}\"
    git submodule sync --recursive
    git submodule update --init --recursive
  fi
  cd \"\${BUILD_DIR}/vision\"
  python -m pip install -r requirements.txt
  export FORCE_CUDA=1
  export USE_ROCM=1
  export PYTORCH_ROCM_ARCH
  export MAX_JOBS
  export CMAKE_PREFIX_PATH=\$(python -c 'import sys; print(sys.prefix)')
  python setup.py develop
fi

python -m pip install -r \"\${REPO_ROOT}/requirements.txt\"
python -m pip install -r \"\${REPO_ROOT}/requirements-gui.txt\"

echo 'ROCm torch check:'
python \"\${REPO_ROOT}/scripts/check_rocm_torch.py\" || true
"
}

install_wrappers() {
  mkdir -p "$HOME/.local/bin"

  cat > "$HOME/.local/bin/iw3-nunif" <<EOF
#!/usr/bin/env bash
set -euo pipefail
BOX="\${NUNIF_BAZZITE_CONTAINER:-$BOX_NAME}"
REPO="\${NUNIF_BAZZITE_REPO:-$REPO_ROOT}"
VENV="\${NUNIF_BAZZITE_VENV:-$VENV_DIR}"
CMD="cd \"\$REPO\" && source \"\$VENV/bin/activate\" && PYTHONPATH=\"\$REPO:\$PYTHONPATH\" python -m iw3 \"\$@\""
exec distrobox enter --name "\$BOX" -- /bin/bash -lc "\$CMD"
EOF
  chmod +x "$HOME/.local/bin/iw3-nunif"

  cat > "$HOME/.local/bin/iw3-gui" <<EOF
#!/usr/bin/env bash
set -euo pipefail
BOX="\${NUNIF_BAZZITE_CONTAINER:-$BOX_NAME}"
REPO="\${NUNIF_BAZZITE_REPO:-$REPO_ROOT}"
VENV="\${NUNIF_BAZZITE_VENV:-$VENV_DIR}"
CMD="cd \"\$REPO\" && source \"\$VENV/bin/activate\" && PYTHONPATH=\"\$REPO:\$PYTHONPATH\" python -m iw3.gui \"\$@\""
exec distrobox enter --name "\$BOX" -- /bin/bash -lc "\$CMD"
EOF
  chmod +x "$HOME/.local/bin/iw3-gui"

  cat > "$HOME/.local/bin/nunif-torch-check" <<EOF
#!/usr/bin/env bash
set -euo pipefail
BOX="\${NUNIF_BAZZITE_CONTAINER:-$BOX_NAME}"
REPO="\${NUNIF_BAZZITE_REPO:-$REPO_ROOT}"
VENV="\${NUNIF_BAZZITE_VENV:-$VENV_DIR}"
CMD="cd \"\$REPO\" && source \"\$VENV/bin/activate\" && PYTHONPATH=\"\$REPO:\$PYTHONPATH\" python scripts/check_rocm_torch.py"
exec distrobox enter --name "\$BOX" -- /bin/bash -lc "\$CMD"
EOF
  chmod +x "$HOME/.local/bin/nunif-torch-check"
}

main() {
  require_fedora_like
  ensure_path
  ensure_distrobox
  create_container
  provision_root
  provision_user
  install_wrappers

  cat <<EOF
Done.
- Wrapper scripts installed to ~/.local/bin: iw3-nunif (CLI), iw3-gui (GUI), nunif-torch-check.
- Container: ${BOX_NAME} (image ${BOX_IMAGE})
- Repo: ${REPO_ROOT}
- Virtualenv: ${VENV_DIR}
Make sure ~/.local/bin is on your PATH, then run:
  iw3-gui
or
  iw3-nunif -h
EOF
}

main "$@"
