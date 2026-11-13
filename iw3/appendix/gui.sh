#!/usr/bin/env bash
###############################################################################
# gui.sh - Launch iw3 GUI on Linux using uv (Python 3.12)
#
# Automatically detects host hardware (NVIDIA CUDA, AMD ROCm, Intel, CPU),
# configures the PyTorch stack and GUI dependencies (wxPython, PyOpenGL, etc.),
# and launches iw3.gui without using the system Python.
###############################################################################
set -e

# Resolve symlinks to find the real script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

# Find nunif project root (supports running from iw3/appendix/, repository root, or external copy)
PROJECT_DIR=""
if [ -n "${NUNIF_DIR:-}" ] && [ -f "$NUNIF_DIR/iw3/gui.py" ]; then
    PROJECT_DIR="$(cd "$NUNIF_DIR" && pwd)"
elif [ -f "$SCRIPT_DIR/requirements.txt" ] && [ -f "$SCRIPT_DIR/iw3/gui.py" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../../requirements.txt" ] && [ -f "$SCRIPT_DIR/../../iw3/gui.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    CHECK_DIR="$SCRIPT_DIR"
    while [ "$CHECK_DIR" != "/" ] && [ -n "$CHECK_DIR" ]; do
        if [ -f "$CHECK_DIR/requirements.txt" ] && [ -f "$CHECK_DIR/iw3/gui.py" ]; then
            PROJECT_DIR="$CHECK_DIR"
            break
        fi
        CHECK_DIR="$(dirname "$CHECK_DIR")"
    done
fi

# Fallback: check common locations if script was copied outside the repository
if [ -z "$PROJECT_DIR" ]; then
    for candidate in \
        "$HOME/github/nunif" \
        "$HOME/nunif" \
        "/home/speed/github/nunif"; do
        if [ -f "$candidate/requirements.txt" ] && [ -f "$candidate/iw3/gui.py" ]; then
            PROJECT_DIR="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi

if [ -z "$PROJECT_DIR" ]; then
    echo "Error: Cannot locate nunif project root." >&2
    echo "Please set NUNIF_DIR=/path/to/nunif environment variable." >&2
    exit 1
fi

# Export PYTHONPATH so python finds iw3 and nunif regardless of current working directory
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PROJECT_DIR}"

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is required but not found in PATH." >&2
    exit 1
fi

# ==============================================================================
# Hardware Auto-Detection & Platform Optimization Engine
# ==============================================================================
GPU_VENDOR="cpu"
GPU_DESC="CPU (Software Fallback)"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

# 1. Check for NVIDIA GPU
if command -v nvidia-smi >/dev/null 2>&1 || [ -e "/dev/nvidia0" ] || (lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iq "nvidia"); then
    GPU_VENDOR="nvidia"
    GPU_DESC="NVIDIA CUDA"
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
        [ -n "$GPU_MODEL" ] && GPU_DESC="NVIDIA ($GPU_MODEL)"
    fi
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 2. Check for AMD ROCm GPU
elif [ -e "/dev/kfd" ] || (lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iqE "amd|advanced micro devices|radeon"); then
    GPU_VENDOR="amd"
    GPU_DESC="AMD ROCm / VA-API"
    AMD_PCI_INFO=$(lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iE "amd|radeon" | head -n 1 || true)
    if [ -n "$AMD_PCI_INFO" ]; then
        GPU_DESC="AMD ($(echo "$AMD_PCI_INFO" | sed -E 's/.*: //'))"
    fi

    # Legacy fallback for ROCm 6.x stacks where gfx1201 was unrecognized
    if [[ "$ROCM_INDEX" =~ rocm6 ]]; then
        if echo "$AMD_PCI_INFO" | grep -iE "Navi 48|RX 9070|RX 90|gfx1200|7550|7551" >/dev/null 2>&1; then
            export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-12.0.0}"
        fi
    fi

    # Do not set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 on consumer RDNA GPUs
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-0}"
    export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-radeonsi}"
    export VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"

    # Preload system libdrm if available to prevent loader conflicts on ROCm
    for libdrm_dir in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
        if [ -f "$libdrm_dir/libdrm.so" ] && [ -f "$libdrm_dir/libdrm_amdgpu.so" ]; then
            export LD_PRELOAD="$libdrm_dir/libdrm.so:$libdrm_dir/libdrm_amdgpu.so${LD_PRELOAD:+:$LD_PRELOAD}"
            break
        fi
    done

# 3. Check for Intel GPU
elif [ -e "/dev/dri/renderD128" ] && (lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iq "intel"); then
    GPU_VENDOR="intel"
    GPU_DESC="Intel VA-API"
    export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-iHD}"
    export VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"
fi

# ==============================================================================
# Configure uv execution environment
# ==============================================================================
UV_RUN_ARGS=(--python 3.12)

# VA-API requires compiling PyAV against host system FFmpeg
if [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "intel" ]; then
    UV_RUN_ARGS+=(
        --no-binary-package av
        --with cython
        --with pkgconfig
    )
fi

# PyTorch dependencies
if [ "$GPU_VENDOR" = "amd" ]; then
    UV_RUN_ARGS+=(--index "$ROCM_INDEX" --with torch --with torchvision)
elif [ "$GPU_VENDOR" = "nvidia" ]; then
    if [ -f "$PROJECT_DIR/requirements-torch.txt" ]; then
        while IFS= read -r url; do
            [ -n "$url" ] && UV_RUN_ARGS+=(--extra-index-url "$url")
        done < <(grep "^--extra-index-url" "$PROJECT_DIR/requirements-torch.txt" | awk '{print $2}')
        UV_RUN_ARGS+=(--with-requirements "$PROJECT_DIR/requirements-torch.txt")
    else
        UV_RUN_ARGS+=(--with torch --with torchvision)
    fi
fi

# Core requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    UV_RUN_ARGS+=(--with-requirements "$PROJECT_DIR/requirements.txt")
fi

# GUI requirements (wxpython, PyOpenGL, python-xlib, mss, etc.)
if [ -f "$PROJECT_DIR/requirements-gui.txt" ]; then
    UV_RUN_ARGS+=(--with-requirements "$PROJECT_DIR/requirements-gui.txt")
fi

# Pre-compiled wxPython GTK3 wheel source for Linux x86_64 (Python 3.12 compatible)
WX_WHEEL_URL="${WX_WHEEL_URL:-https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04/}"
if [ -n "$WX_WHEEL_URL" ]; then
    UV_RUN_ARGS+=(--find-links "$WX_WHEEL_URL")
fi

# Do not attempt to build wxPython from sdist (PyPI only ships Linux source distributions,
# which requires hours to compile wxWidgets and fails without full GTK3 dev packages;
# use the precompiled binary wheel from --find-links instead)
UV_RUN_ARGS+=(--no-build-package wxpython)

if [ -n "${UV_OFFLINE:-}" ]; then
    UV_RUN_ARGS+=(--offline)
fi

echo "==> Launching iw3 GUI via uv"
echo "    Platform: $GPU_DESC"
echo "    Python:   3.12 (managed by uv)"
echo "    Root:     $PROJECT_DIR"

if [ "${1:-}" = "--dry-run" ]; then
    echo "Dry run - command to execute:"
    printf 'uv run'
    printf ' %q' "${UV_RUN_ARGS[@]}"
    printf ' python -m iw3.gui\n'
    exit 0
fi

cd "$PROJECT_DIR"
exec uv run "${UV_RUN_ARGS[@]}" python -m iw3.gui "$@"
