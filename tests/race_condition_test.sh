#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.."; pwd)
cd "${PROJECT_DIR}"

# Generate test data if not exists
bash "${SCRIPT_DIR}/race_condition_data.sh"

SD_VIDEO="tests/data/race_condition/sd.mkv"
HDR_VIDEO="tests/data/race_condition/hdr.mkv"
SD_NOAUDIO_VIDEO="tests/data/race_condition/sd_noaudio.mkv"
HDR_NOAUDIO_VIDEO="tests/data/race_condition/hdr_noaudio.mkv"

WORK_DIR="tests/data/race_condition_test"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

COMMON_REF_ARGS="--low-vram --scene-detect --ema-normalize --ema-buffer 30 -y"
COMMON_TEST_ARGS="--batch-size 4 --max-workers 2 --cuda-stream --scene-detect --ema-normalize --ema-buffer 30 -y"

run_and_compare() {
    local test_name="$1"
    local input_video="$2"
    local extra_args="$3"

    echo "======================================================================"
    echo "Running Test: ${test_name}"
    echo "Input: ${input_video}"
    echo "Extra args: ${extra_args}"
    echo "======================================================================"

    local ref_output="${WORK_DIR}/${test_name}_ref.mkv"
    local test_output="${WORK_DIR}/${test_name}_test.mkv"

    # Run reference (sequential, low-vram)
    python -m iw3.cli -i "${input_video}" -o "${ref_output}" ${COMMON_REF_ARGS} ${extra_args}

    # Run test target (batch, multi-worker, cuda-stream)
    python -m iw3.cli -i "${input_video}" -o "${test_output}" ${COMMON_TEST_ARGS} ${extra_args}

    # Compare with PSNR
    local stats
    stats=$(ffmpeg -i "${ref_output}" -i "${test_output}" -lavfi psnr -f null - 2>&1 | grep "PSNR y:" || true)
    echo "PSNR output: ${stats}"

    if echo "${stats}" | grep -q "average:inf"; then
        echo "[PASS] ${test_name}: PSNR inf (Bit-identical)"
    else
        local avg_psnr
        avg_psnr=$(echo "${stats}" | sed -E 's/.*average:([0-9.]+).*/\1/')
        if [ -n "${avg_psnr}" ] && (( $(echo "${avg_psnr} >= 38" | bc -l) )); then
            echo "[PASS] ${test_name}: PSNR ${avg_psnr} >= 38"
        else
            echo "[FAIL] ${test_name}: PSNR ${avg_psnr:-none} < 38"
            return 1
        fi
    fi
}

FAILED=0

# 1. VDA_S + sd.mkv (uint8) + libx265 (CPU encode, primary bug reproduction case)
run_and_compare "vda_s_sd_libx265" "${SD_VIDEO}" "--depth-model VDA_S --video-codec libx265" || FAILED=1

# 2. VDA_S + sd.mkv (uint8) + hevc_nvenc (GPU encode)
run_and_compare "vda_s_sd_nvenc" "${SD_VIDEO}" "--depth-model VDA_S --video-codec hevc_nvenc" || FAILED=1

# 3. VDA_S + hdr.mkv (float16) + libx265 (HDR / 10bit)
run_and_compare "vda_s_hdr_libx265" "${HDR_VIDEO}" "--depth-model VDA_S --video-codec libx265 --pix-fmt yuv420p10le" || FAILED=1

# 4. Any_S + sd.mkv (uint8) + libx265 (WorkerPool queue / CPU encode)
run_and_compare "any_s_sd_libx265" "${SD_VIDEO}" "--depth-model Any_S --video-codec libx265" || FAILED=1

# 5. Any_S + sd.mkv (uint8) + hevc_nvenc (WorkerPool queue / GPU encode)
run_and_compare "any_s_sd_nvenc" "${SD_VIDEO}" "--depth-model Any_S --video-codec hevc_nvenc" || FAILED=1

# 6. Any_S + hdr.mkv (float16) + hevc_nvenc (WorkerPool queue / HDR / 10bit)
run_and_compare "any_s_hdr_nvenc" "${HDR_VIDEO}" "--depth-model Any_S --video-codec hevc_nvenc --pix-fmt yuv420p10le" || FAILED=1

# 7. Any_S + sd.mkv + forward_inpaint (Inpaint pipeline async path)
run_and_compare "any_s_sd_inpaint" "${SD_VIDEO}" "--depth-model Any_S --method forward_inpaint --video-codec libx265" || FAILED=1

# 8. VDA_S + sd_noaudio.mkv + hevc_nvenc (no-audio GPU encode bug test)
run_and_compare "vda_s_sd_nvenc_noaudio" "${SD_NOAUDIO_VIDEO}" "--depth-model VDA_S --video-codec hevc_nvenc" || FAILED=1

# 9. Any_S + sd_noaudio.mkv + hevc_nvenc (no-audio WorkerPool + GPU encode test)
run_and_compare "any_s_sd_nvenc_noaudio" "${SD_NOAUDIO_VIDEO}" "--depth-model Any_S --video-codec hevc_nvenc" || FAILED=1

# 10. VDA_S + sd_noaudio.mkv + libx265 (no-audio CPU encode test)
run_and_compare "vda_s_sd_libx265_noaudio" "${SD_NOAUDIO_VIDEO}" "--depth-model VDA_S --video-codec libx265" || FAILED=1

# Cleanup temp work dir if all passed
if [ "${FAILED}" -eq 0 ]; then
    echo "======================================================================"
    echo "ALL TESTS PASSED!"
    echo "======================================================================"
    rm -rf "${WORK_DIR}"
    exit 0
else
    echo "======================================================================"
    echo "SOME TESTS FAILED! (Check files in ${WORK_DIR})"
    echo "======================================================================"
    exit 1
fi
