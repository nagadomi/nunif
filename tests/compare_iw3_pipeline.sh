#!/bin/bash


if [ "$#" != 3 ]; then
    echo "usage: $0 <repo1_path> <repo2_path> <test_video_path>"
    echo "example: ./tests/compare_iw3_pipeline.sh ../nunif-old ./ ./tmp/test_videos/4k_hlg_5sec.mkv"
    exit 1
fi

SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
REPO1_PATH=$(realpath "$1")
REPO2_PATH=$(realpath "$2")
VIDEO_PATH=$(realpath "$3")

OUTPUT1_DIR="${SCRIPT_DIR}/data/compare_iw3_pipeline/repo1"
OUTPUT2_DIR="${SCRIPT_DIR}/data/compare_iw3_pipeline/repo2"
CACHE_DIR="${SCRIPT_DIR}/data/compare_iw3_pipeline/cache"

CURRENT_AV_VERSION=$(pip show av | grep Version | awk '{print $2}')
REPO1_AV="av==15.1.0"
REPO2_AV="av==17.0.1"
COMMON_OPTIONS="--video-codec hevc_nvenc"


rm -rf "${OUTPUT1_DIR}" "${OUTPUT2_DIR}"
mkdir -p "${OUTPUT1_DIR}"
mkdir -p "${OUTPUT2_DIR}"

REPOS=("${REPO1_PATH}" "${REPO2_PATH}")
OUTPUTS=("${OUTPUT1_DIR}" "${OUTPUT2_DIR}")
AVS=("${REPO1_AV}" "${REPO2_AV}")

for i in "${!REPOS[@]}"
do
    repo_path="${REPOS[$i]}"
    output_dir="${OUTPUTS[$i]}"
    av="${AVS[$i]}"
    if [ "$i" -eq 1 ]; then
        extra_args="--hwaccel cuda"
    fi

    cd "${repo_path}" || continue

    set -x

    # update av
    pip3 install ${av}

    output_file="${output_dir}/batch.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model Any_V2_S --max-workers 2 --batch-size 2 ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/batch_cuda.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model Any_V2_S --max-workers 2 --batch-size 2 --cuda-stream ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/low_vram.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model Any_V2_S --low-vram ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/batch_ema.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model Any_V2_S --max-workers 2 --batch-size 2 --cuda-stream --ema-normalize ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/low_vram_ema.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model Any_V2_S --low-vram --ema-normalize ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/vda.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model VDA_S --ema-normalize --batch-size 2 --scene-detect --scene-cache-dir ${CACHE_DIR} ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/inpaint_batch.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 30 --batch-size 2 --scene-detect --scene-cache-dir ${CACHE_DIR} ${COMMON_OPTIONS} ${extra_args}

    output_file="${output_dir}/inpaint_vda.mkv"
    python -m iw3.cli -y -i ${VIDEO_PATH} -o ${output_file} --depth-model VDA_S  --method mlbw_l2_inpaint --ema-normalize  --ema-buffer 30 --batch-size 2 --scene-detect --scene-cache-dir ${CACHE_DIR} ${COMMON_OPTIONS} ${extra_args}

    set +x
done


output_files=(
    batch.mkv
    batch_cuda.mkv
    low_vram.mkv
    batch_ema.mkv
    low_vram_ema.mkv
    vda.mkv
    inpaint_batch.mkv
    inpaint_vda.mkv
)
for filename in "${output_files[@]}"
do
    file1="${OUTPUT1_DIR}/${filename}"
    file2="${OUTPUT2_DIR}/${filename}"

    echo "--- Comparing: $filename ---"

    if [[ ! -f "$file1" || ! -f "$file2" ]]; then
        echo "  [!!] Error: Output file missing."
        continue
    fi

    stats=$(ffmpeg -i "$file1" -i "$file2" -lavfi psnr -f null - 2>&1 | grep "PSNR y:")
    avg_psnr=$(echo "$stats" | sed -E 's/.*average:([0-9.]+).*/\1/')

    if [ -z "$avg_psnr" ]; then
        echo "  [!!] Error: Could not calculate PSNR (possible decode error)"
    elif (( $(echo "$avg_psnr >= 38" | bc -l) )); then
        echo "  [OK] Quality Pass: PSNR $avg_psnr"
    else
        echo "  [FAIL] Quality Low: PSNR $avg_psnr"
    fi
done


pip3 install "av==${CURRENT_AV_VERSION}"
