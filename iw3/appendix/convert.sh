#!/usr/bin/env bash
###############################################################################
# convert.sh - Universal Video to 3D SBS / VR180 Conversion Script using iw3 & uv
#
# Automatically detects host hardware (NVIDIA CUDA, AMD ROCm, Intel, CPU)
# and configures the optimal PyTorch stack, hardware decoder, and video encoder.
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
if [ -n "${NUNIF_DIR:-}" ] && [ -f "$NUNIF_DIR/iw3/cli.py" ]; then
    PROJECT_DIR="$(cd "$NUNIF_DIR" && pwd)"
elif [ -f "$SCRIPT_DIR/requirements.txt" ] && [ -f "$SCRIPT_DIR/iw3/cli.py" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../../requirements.txt" ] && [ -f "$SCRIPT_DIR/../../iw3/cli.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    # Traverse parent directories of script location
    CHECK_DIR="$SCRIPT_DIR"
    while [ "$CHECK_DIR" != "/" ] && [ -n "$CHECK_DIR" ]; do
        if [ -f "$CHECK_DIR/requirements.txt" ] && [ -f "$CHECK_DIR/iw3/cli.py" ]; then
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
        if [ -f "$candidate/requirements.txt" ] && [ -f "$candidate/iw3/cli.py" ]; then
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
DEFAULT_CODEC="libx264"
DEFAULT_HWACCEL=""
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
    DEFAULT_CODEC="hevc_nvenc"
    DEFAULT_HWACCEL="cuda"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 2. Check for AMD ROCm GPU
elif [ -e "/dev/kfd" ] || (lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iqE "amd|advanced micro devices|radeon"); then
    GPU_VENDOR="amd"
    GPU_DESC="AMD ROCm / VA-API"
    AMD_PCI_INFO=$(lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iE "amd|radeon" | head -n 1 || true)
    if [ -n "$AMD_PCI_INFO" ]; then
        GPU_DESC="AMD ($(echo "$AMD_PCI_INFO" | sed -E 's/.*: //'))"
    fi

    # WARNING: Do NOT unconditionally force HSA_OVERRIDE_GFX_VERSION=12.0.0 on ROCm 7+.
    # Navi 48 (RX 9070/XT) is gfx1201; forcing 12.0.0 (Navi 44) dispatches mismatched
    # code objects and causes kernel page faults (TCP 0x8) and display lockups.
    # Only keep 12.0.0 as a legacy fallback for ROCm 6.x stacks where gfx1201 was unrecognized.
    if [[ "$ROCM_INDEX" =~ rocm6 ]]; then
        if echo "$AMD_PCI_INFO" | grep -iE "Navi 48|RX 9070|RX 90|gfx1200|7550|7551" >/dev/null 2>&1; then
            export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-12.0.0}"
        fi
    fi

    # WARNING: Do NOT set PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True
    # on AMD. Dynamic virtual address unmapping in HIPCachingAllocator causes amdgpu GCVM_L2
    # page faults, display artifacts, and driver crashes when VRAM is shared with VA-API / compositor.

    # WARNING: Do NOT set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 on consumer RDNA GPUs.
    # Experimental AOTriton attention kernels can hang the GPU compute ring.
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

    DEFAULT_CODEC="hevc_vaapi"
    DEFAULT_HWACCEL="vaapi"

# 3. Check for Intel GPU
elif [ -e "/dev/dri/renderD128" ] && (lspci 2>/dev/null | grep -iE "vga|3d|display" | grep -iq "intel"); then
    GPU_VENDOR="intel"
    GPU_DESC="Intel VA-API"
    export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-iHD}"
    export VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"
    DEFAULT_CODEC="hevc_vaapi"
    DEFAULT_HWACCEL="vaapi"
fi

# ==============================================================================
# Pipeline Defaults
# ==============================================================================
DEFAULT_DEPTH_MODEL="Any_V2_S"
DEFAULT_METHOD="row_flow"
DEFAULT_BATCH_SIZE=4
DEFAULT_RESOLUTION=518
DEFAULT_VR180=1
DEFAULT_CRF=22
DEFAULT_EMA_NORMALIZE=1
DEFAULT_DEPTH_AA=1

# Hardware bounds for VR180 / hardware encoders (prevents GPU page faults on 4K+ inputs)
DEFAULT_MAX_OUTPUT_WIDTH="5760"
DEFAULT_MAX_OUTPUT_HEIGHT="2880"
USER_MAX_OUTPUT_WIDTH=""
USER_MAX_OUTPUT_HEIGHT=""

DEPTH_MODEL="$DEFAULT_DEPTH_MODEL"
METHOD="$DEFAULT_METHOD"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
RESOLUTION="$DEFAULT_RESOLUTION"
VR180="$DEFAULT_VR180"
CRF="$DEFAULT_CRF"
EMA_NORMALIZE="$DEFAULT_EMA_NORMALIZE"
DEPTH_AA="$DEFAULT_DEPTH_AA"
VIDEO_CODEC="$DEFAULT_CODEC"
HWACCEL="$DEFAULT_HWACCEL"
INPUT_FILE=""
OUTPUT_FILE=""

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] <input_video> [output_video]
       $(basename "$0") input.mp4 [output.mp4]

Universal Single-Pass SSD-Safe 2D-to-VR180/3D Conversion Pipeline:
  Streams frames directly in GPU/VRAM from hardware decoding through depth estimation,
  stereo warping, and hardware video encoding. Zero intermediate PNG files are written
  to disk, protecting your SSD and maximizing pipeline throughput.

Auto-Detected Hardware Platform:
  Platform:       ${GPU_DESC}
  Default Codec:  ${DEFAULT_CODEC}
  Default HWAccel:${DEFAULT_HWACCEL:-None (Software)}

Active Defaults:
  Depth Model:    ${DEFAULT_DEPTH_MODEL}
  Method:         ${DEFAULT_METHOD}
  Batch Size:     ${DEFAULT_BATCH_SIZE}
  Resolution:     ${DEFAULT_RESOLUTION}
  Format:         VR180 (--vr180 enabled)
  Quality (CRF):  ${DEFAULT_CRF}
  EMA Normalize:  Enabled (--ema-normalize)
  Depth AA:       Enabled (--depth-aa)

Pipeline Configuration & Tuning Reference:
  - Depth Models:
      Any_V2_S    : Depth-Anything-V2 Small (Fastest, low VRAM, clean edges; default)
      Any_V2_B    : Depth-Anything-V2 Base (Higher accuracy and edge fidelity)
      ZoeD_Any_N  : Metric depth tuned for indoor and natural human scenes
      VDA_B / L   : Video Depth Anything (strong temporal consistency across frames)
  - Depth Resolutions:
      518         : High throughput. Softer edge gradients prevent clothing/collar tearing.
                    Fits comfortably inside GPU cache (e.g. AMD 64MB Infinity Cache).
      756         : Medium resolution. Sharper detail; pair with mlbw_l4.
      1008        : Maximum depth precision for 4K video. Computationally heavy.
  - Stereo Synthesis Methods:
      mlbw_l2     : 2-layer multi-layer backward warp. Balanced quality & speed (default).
      mlbw_l4     : 4-layer multi-layer backward warp. Flawless edge separation on fine
                    boundaries with zero disocclusion tearing.
      mlbw_l4s    : Lightweight/faster variant of mlbw_l4.
      row_flow    : Fast row-based disparity neural flow. Clean on smooth contours.
      forward     : Forward warping alternative with hole filling.
  - Batch Size:
      4           : Sweet spot for saturating GPU compute while staying inside cache/VRAM.
      8           : High throughput for lighter models (518 / row_flow).
  - Temporal Smoothing & Boundary Refinement:
      --depth-aa  : Learned FP32 depth boundary refinement model; smooths pixel stair-stepping.
      --ema-normalize / --no-ema:
                    Flicker reduction; temporal exponential moving average smoothing over
                    depth min/max to eliminate 3D depth pulsing across frames.

Positional & File Arguments:
  <input_video>                 Input 2D video file (required, or pass -i, --input)
  [output_video]                Output 3D video file (optional, or pass -o, --output)
                                Default: <input>_3d.<ext>

Depth & Synthesis:
  -m, --depth-model <name>      Depth model (default: ${DEFAULT_DEPTH_MODEL})
  --method <method>             Synthesis method (default: ${DEFAULT_METHOD})
                                Choices: mlbw_l2, mlbw_l4, mlbw_l4s, row_flow, forward, etc.
  -r, --resolution <int>        Resolution (small side) for depth model (default: ${DEFAULT_RESOLUTION})
  --no-resolution               Disable explicit resolution override
  -b, --batch-size <int>        Batch size for inference (default: ${DEFAULT_BATCH_SIZE})
  -d, --divergence <float>      3D divergence / effect strength (default: 2.0)
  -c, --convergence <float>     Convergence plane distance 0-1 (default: 0.5)
  --depth-aa                    Enable depth antialiasing (default: enabled)
  --no-depth-aa                 Disable depth antialiasing
  --ema-normalize               EMA depth normalization across frames (default: enabled)
  --no-ema, --no-ema-normalize  Disable EMA depth normalization
  --scene-detect                Shot boundary detection for EMA resets
  --tta                         Enable test-time flip augmentation

3D Output Format:
  --vr180                       Output in VR180 equirectangular format (default: enabled)
  --no-vr180                    Disable VR180 output (outputs Full SBS)
  --half-sbs                    Output in Half Side-by-Side
  --tb                          Output in Full Top-Bottom
  --half-tb                     Output in Half Top-Bottom
  --anaglyph [MODE]             Output in Anaglyph 3D (dubois, color, gray, etc.)
  --ipd-offset <float>          IPD offset in % (default: 0)

Video Quality & Encoding:
  --crf <int>                   Video quality CRF/QP (default: ${DEFAULT_CRF}, lower = higher quality)
  --fps, --max-fps <float>      Limit output framerate (default: 30)
  -vc, --video-codec <codec>    Video codec (auto-detected: ${DEFAULT_CODEC})
                                Choices: hevc_nvenc, h264_nvenc, hevc_vaapi, h264_vaapi, libx265, libx264
  --preset <preset>             Encoder preset (e.g. ultrafast, fast, medium, slow, p1-p7)
  --pix-fmt <format>            Pixel format (default: yuv420p)
  --max-output-width <int>      Limit output video width
  --max-output-height <int>     Limit output video height

Hardware & Performance:
  -g, --gpu <id>                GPU device ID (-1 for CPU, default: 0)
  --hwaccel <accel>             Hardware accelerator for video decoder (auto-detected: ${DEFAULT_HWACCEL:-None})
                                Choices: cuda, vaapi, none
  --no-hwaccel                  Disable hardware decoding (forces software decoder)
  --vaapi-device <device>       VAAPI render node (default: /dev/dri/renderD128)
  --disable-software-fallback   Fail if hardware decoding fails rather than falling back to CPU
  --low-vram                    Disable batching for low memory GPUs
  --cuda-stream                 Use multi CUDA streams
  --compile                     Compile models with PyTorch 2.x
  --max-workers <int>           Inference worker threads (0 to 16, default: 0)

General:
  -i, --input <file>            Explicit input video path
  -o, --output <file>           Explicit output video path
  --start-time <time>           Start time offset (hh:mm:ss or mm:ss or seconds)
  --end-time <time>             End time offset (hh:mm:ss or mm:ss or seconds)
  -y, --yes                     Overwrite output file without asking
  --resume                      Skip processing if output file already exists
  -h, --help                    Show this help message and exit

Examples:
  # Universal auto-detect (runs optimal hardware decoding + encoding):
  $(basename "$0") input.mp4
  $(basename "$0") input.mp4 output_3d.mp4 --fps 60 --crf 20

  # Trim clip and custom resolution:
  $(basename "$0") input.mp4 --end-time 00:00:30 --resolution 518 --batch-size 8

  # High-quality stereo warping:
  $(basename "$0") input.mp4 --method mlbw_l4 --depth-model Any_V2_S

  # Explicit NVIDIA NVENC / CUDA override:
  $(basename "$0") input.mp4 -vc hevc_nvenc --hwaccel cuda

  # Explicit AMD VA-API override:
  $(basename "$0") input.mp4 -vc hevc_vaapi --hwaccel vaapi
EOF
}

# When called without any parameters or with help flag, list options
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

POSITIONAL=()
CLI_ARGS=()
USER_CODEC=""
USER_HWACCEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --method=*)
            METHOD="${1#*=}"
            shift 1
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -b=*|--batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift 1
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -r=*|--resolution=*)
            RESOLUTION="${1#*=}"
            shift 1
            ;;
        --no-resolution)
            RESOLUTION=""
            shift 1
            ;;
        -m|--depth-model|--model)
            DEPTH_MODEL="$2"
            shift 2
            ;;
        -m=*|--depth-model=*|--model=*)
            DEPTH_MODEL="${1#*=}"
            shift 1
            ;;
        --crf)
            CRF="$2"
            shift 2
            ;;
        --crf=*)
            CRF="${1#*=}"
            shift 1
            ;;
        --fps|--max-fps)
            CLI_ARGS+=(--max-fps "$2")
            shift 2
            ;;
        --fps=*|--max-fps=*)
            CLI_ARGS+=(--max-fps "${1#*=}")
            shift 1
            ;;
        --vr180)
            VR180=1
            shift 1
            ;;
        --no-vr180)
            VR180=0
            shift 1
            ;;
        --half-sbs|--tb|--half-tb)
            VR180=0
            CLI_ARGS+=("$1")
            shift 1
            ;;
        --anaglyph)
            VR180=0
            if [[ $# -ge 2 && ! "$2" =~ ^- ]]; then
                CLI_ARGS+=("$1" "$2")
                shift 2
            else
                CLI_ARGS+=("$1")
                shift 1
            fi
            ;;
        --ema-normalize)
            EMA_NORMALIZE=1
            shift 1
            ;;
        --no-ema-normalize|--no-ema)
            EMA_NORMALIZE=0
            shift 1
            ;;
        --depth-aa)
            DEPTH_AA=1
            shift 1
            ;;
        --no-depth-aa)
            DEPTH_AA=0
            shift 1
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -i=*|--input=*)
            INPUT_FILE="${1#*=}"
            shift 1
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -o=*|--output=*)
            OUTPUT_FILE="${1#*=}"
            shift 1
            ;;
        -vc|--video-codec)
            USER_CODEC="$2"
            shift 2
            ;;
        -vc=*|--video-codec=*)
            USER_CODEC="${1#*=}"
            shift 1
            ;;
        --hwaccel)
            USER_HWACCEL="$2"
            shift 2
            ;;
        --hwaccel=*)
            USER_HWACCEL="${1#*=}"
            shift 1
            ;;
        --no-hwaccel)
            USER_HWACCEL="none"
            shift 1
            ;;
        -d|--divergence)
            CLI_ARGS+=(--divergence "$2")
            shift 2
            ;;
        -d=*|--divergence=*)
            CLI_ARGS+=(--divergence "${1#*=}")
            shift 1
            ;;
        -c|--convergence)
            CLI_ARGS+=(--convergence "$2")
            shift 2
            ;;
        -c=*|--convergence=*)
            CLI_ARGS+=(--convergence "${1#*=}")
            shift 1
            ;;
        -g|--gpu)
            CLI_ARGS+=(--gpu "$2")
            shift 2
            ;;
        -g=*|--gpu=*)
            CLI_ARGS+=(--gpu "${1#*=}")
            shift 1
            ;;
        --vaapi-device)
            CLI_ARGS+=(--vaapi-device "$2")
            shift 2
            ;;
        --vaapi-device=*)
            CLI_ARGS+=(--vaapi-device "${1#*=}")
            shift 1
            ;;
        --disable-software-fallback)
            CLI_ARGS+=("$1")
            shift 1
            ;;
        --start-time)
            CLI_ARGS+=(--start-time "$2")
            shift 2
            ;;
        --start-time=*)
            CLI_ARGS+=(--start-time "${1#*=}")
            shift 1
            ;;
        --end-time)
            CLI_ARGS+=(--end-time "$2")
            shift 2
            ;;
        --end-time=*)
            CLI_ARGS+=(--end-time "${1#*=}")
            shift 1
            ;;
        -y|--yes)
            CLI_ARGS+=(--yes)
            shift 1
            ;;
        --tta|--low-vram|--cuda-stream|--compile|--scene-detect|--resume)
            CLI_ARGS+=("$1")
            shift 1
            ;;
        --max-output-width)
            USER_MAX_OUTPUT_WIDTH="$2"
            shift 2
            ;;
        --max-output-width=*)
            USER_MAX_OUTPUT_WIDTH="${1#*=}"
            shift 1
            ;;
        --max-output-height)
            USER_MAX_OUTPUT_HEIGHT="$2"
            shift 2
            ;;
        --max-output-height=*)
            USER_MAX_OUTPUT_HEIGHT="${1#*=}"
            shift 1
            ;;
        --preset|--pix-fmt|--max-workers|--ipd-offset|--ema-decay|--autocrop|--edge-dilation)
            if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
                CLI_ARGS+=("$1" "$2")
                shift 2
            else
                CLI_ARGS+=("$1")
                shift 1
            fi
            ;;
        --*=*)
            CLI_ARGS+=("$1")
            shift 1
            ;;
        -*)
            # Passthrough any other iw3 flags
            if [[ $# -ge 2 && ! "$2" =~ ^- ]]; then
                CLI_ARGS+=("$1" "$2")
                shift 2
            else
                CLI_ARGS+=("$1")
                shift 1
            fi
            ;;
        *)
            POSITIONAL+=("$1")
            shift 1
            ;;
    esac
done

# Resolve input and output file paths
if [ -z "$INPUT_FILE" ] && [ ${#POSITIONAL[@]} -ge 1 ]; then
    INPUT_FILE="${POSITIONAL[0]}"
    if [ -z "$OUTPUT_FILE" ] && [ ${#POSITIONAL[@]} -ge 2 ]; then
        OUTPUT_FILE="${POSITIONAL[1]}"
    fi
fi

if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input video specified." >&2
    echo "" >&2
    show_usage >&2
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found." >&2
    exit 1
fi

if [ -z "$OUTPUT_FILE" ]; then
    DIR="$(dirname "$INPUT_FILE")"
    BASE="$(basename "$INPUT_FILE")"
    STEM="${BASE%.*}"
    EXT="${BASE##*.}"
    OUTPUT_FILE="${DIR}/${STEM}_3d.${EXT}"
fi

FINAL_ARGS=()

if [ -n "$DEPTH_MODEL" ]; then
    FINAL_ARGS+=(--depth-model "$DEPTH_MODEL")
fi
if [ -n "$METHOD" ]; then
    FINAL_ARGS+=(--method "$METHOD")
fi
if [ -n "$BATCH_SIZE" ]; then
    FINAL_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [ -n "$RESOLUTION" ]; then
    FINAL_ARGS+=(--resolution "$RESOLUTION")
fi
if [ "$VR180" -eq 1 ]; then
    FINAL_ARGS+=(--vr180)
fi
if [ -n "$CRF" ]; then
    FINAL_ARGS+=(--crf "$CRF")
fi
if [ "$EMA_NORMALIZE" -eq 1 ]; then
    FINAL_ARGS+=(--ema-normalize)
fi
if [ "$DEPTH_AA" -eq 1 ]; then
    FINAL_ARGS+=(--depth-aa)
fi

# Apply safe output resolution bounds for VR180 / hardware encoders
if [ "$VR180" -eq 1 ]; then
    FINAL_ARGS+=(--max-output-width "${USER_MAX_OUTPUT_WIDTH:-$DEFAULT_MAX_OUTPUT_WIDTH}")
    FINAL_ARGS+=(--max-output-height "${USER_MAX_OUTPUT_HEIGHT:-$DEFAULT_MAX_OUTPUT_HEIGHT}")
elif [ -n "$USER_MAX_OUTPUT_WIDTH" ] || [ -n "$USER_MAX_OUTPUT_HEIGHT" ]; then
    [ -n "$USER_MAX_OUTPUT_WIDTH" ] && FINAL_ARGS+=(--max-output-width "$USER_MAX_OUTPUT_WIDTH")
    [ -n "$USER_MAX_OUTPUT_HEIGHT" ] && FINAL_ARGS+=(--max-output-height "$USER_MAX_OUTPUT_HEIGHT")
fi

# Apply video codec (user override takes precedence over auto-detected default)
ACTIVE_CODEC="${USER_CODEC:-$DEFAULT_CODEC}"
if [ -n "$ACTIVE_CODEC" ]; then
    FINAL_ARGS+=(--video-codec "$ACTIVE_CODEC")
fi

# Apply hardware decoder acceleration
ACTIVE_HWACCEL="${USER_HWACCEL:-$DEFAULT_HWACCEL}"
if [ -n "$ACTIVE_HWACCEL" ] && [ "$ACTIVE_HWACCEL" != "none" ]; then
    FINAL_ARGS+=(--hwaccel "$ACTIVE_HWACCEL")
    if [ "$ACTIVE_HWACCEL" = "vaapi" ] && [ -n "${VAAPI_DEVICE:-}" ]; then
        FINAL_ARGS+=(--vaapi-device "$VAAPI_DEVICE")
    fi
fi

# Configure uv run flags: uv manages Python 3.12 and dependencies autonomously
UV_RUN_ARGS=(--python 3.12)

# VA-API requires compiling PyAV against host system FFmpeg; NVIDIA pre-built wheel has CUDA & NVENC out of the box
if [ "$ACTIVE_HWACCEL" = "vaapi" ] || [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "intel" ]; then
    UV_RUN_ARGS+=(
        --no-binary-package av
        --with cython
        --with pkgconfig
    )
fi

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

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    UV_RUN_ARGS+=(--with-requirements "$PROJECT_DIR/requirements.txt")
fi

if [ -n "${UV_OFFLINE:-}" ]; then
    UV_RUN_ARGS+=(--offline)
fi

exec uv run "${UV_RUN_ARGS[@]}" python -m iw3.cli --input "$INPUT_FILE" --output "$OUTPUT_FILE" "${FINAL_ARGS[@]}" "${CLI_ARGS[@]}"
