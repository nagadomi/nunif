#!/bin/bash

# For comparison with results from the old repository (2026-04-23), dev vs av17 branch.
# NOTE:
#   In av17, bt601 was changed from yuv420p(tv, bt470bg/bt470bg/smpte170m) to yuv420p(tv, bt470bg/smpte170m/smpte170m), so the bt601 metadata no longer matches.
#   In the old branch, bt709-pc/bt601-pc are encoded as yuvj420p, which causes an error with NVENC, so it cannot be compared.
#   In av17, HDR-to-SDR conversion was reworked to use LUT, so PSNR is not higher, but the results are better in av17.

if [ "$#" != 3 ]; then
    echo "usage: $0 <repo1_path> <repo2_path> <test_video_path>"
    echo "example: ./tests/compare_color_transform.sh ../nunif-old ./ ./tmp/test_videos/4k_hlg_5sec.mkv"
    exit 1
fi

SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
REPO1_PATH=$(realpath "$1")
REPO2_PATH=$(realpath "$2")
VIDEO_PATH=$(realpath "$3")

OUTPUT1_DIR="${SCRIPT_DIR}/data/compare_color_transform/repo1"
OUTPUT2_DIR="${SCRIPT_DIR}/data/compare_color_transform/repo2"

CURRENT_AV_VERSION=$(pip show av | grep Version | awk '{print $2}')
REPO1_AV="av==15.1.0"
REPO2_AV="av==17.0.1"


IW3_OPTIONS=(-i "${VIDEO_PATH}" --yes --half-sbs --method row_flow_v3 --depth-model Any_V2_S --batch-size 2 --max-workers 2 --cuda-stream)


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

    cd "${repo_path}" || continue

    # update av
    pip3 install ${av}

    for encoder in libx265 hevc_nvenc
    do
        for pix_fmt in yuv420p yuv420p10le
        do
            for colorspace in auto bt709-tv bt709-pc bt601-tv
            do
                if [ "$encoder" = "hevc_nvenc" ] && [ "$colorspace" = "bt709-pc" ]; then
                    continue
                fi
                output_file="${output_dir}/${encoder}_${pix_fmt}_${colorspace}.mkv"
                echo "--------------------------------------------------"
                echo "Processing: ${output_file}"
                python -m iw3.cli "${IW3_OPTIONS[@]}" \
                    --video-codec "${encoder}" \
                    --pix-fmt "${pix_fmt}" \
                    --colorspace "${colorspace}" \
                    -o "${output_file}"

                output_file="${output_dir}/${encoder}_${pix_fmt}_${colorspace}_vf.mkv"
                python -m iw3.cli "${IW3_OPTIONS[@]}" \
                    --video-codec "${encoder}" \
                    --pix-fmt "${pix_fmt}" \
                    --colorspace "${colorspace}" \
                    --vf crop=x=0:y=0:w=iw:h=ih \
                    -o "${output_file}"

                if [ "${repo_path}" == ${REPO2_PATH} ]; then
                    output_file="${output_dir}/${encoder}_${pix_fmt}_${colorspace}_cuda.mkv"
                    echo "Processing: ${output_file}"
                    python -m iw3.cli "${IW3_OPTIONS[@]}" \
                           --video-codec "${encoder}" \
                           --pix-fmt "${pix_fmt}" \
                           --colorspace "${colorspace}" \
                           --hwaccel cuda \
                           -o "${output_file}"

                    output_file="${output_dir}/${encoder}_${pix_fmt}_${colorspace}_cuda_vf.mkv"
                    echo "Processing: ${output_file}"
                    python -m iw3.cli "${IW3_OPTIONS[@]}" \
                           --video-codec "${encoder}" \
                           --pix-fmt "${pix_fmt}" \
                           --colorspace "${colorspace}" \
                           --hwaccel cuda \
                           --vf crop=x=0:y=0:w=iw:h=ih \
                           -o "${output_file}"
                fi
            done
        done
    done
done


# Compare
for encoder in libx265 hevc_nvenc
do
    for pix_fmt in yuv420p yuv420p10le
    do
        for colorspace in auto bt709-tv bt709-pc bt601-tv
        do
            output1="${OUTPUTS[0]}/${encoder}_${pix_fmt}_${colorspace}.mkv"
            output2="${OUTPUTS[1]}/${encoder}_${pix_fmt}_${colorspace}.mkv"
            output3="${OUTPUTS[1]}/${encoder}_${pix_fmt}_${colorspace}_cuda.mkv"

            output1_vf="${OUTPUTS[0]}/${encoder}_${pix_fmt}_${colorspace}_vf.mkv"
            output2_vf="${OUTPUTS[1]}/${encoder}_${pix_fmt}_${colorspace}_vf.mkv"
            output3_vf="${OUTPUTS[1]}/${encoder}_${pix_fmt}_${colorspace}_cuda_vf.mkv"

            if [ ! -f "$output1" ] || [ ! -f "$output2" ]; then
                echo "Skip: File not found ($encoder $pix_fmt $colorspace)"
                continue
            fi

            echo "--------------------------------------------------"
            echo "Comparing: ${encoder} / ${pix_fmt} / ${colorspace}"

            meta1=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output1")
            meta2=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output2")
            meta3=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output3")

            meta1_vf=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output1_vf")
            meta2_vf=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output2_vf")
            meta3_vf=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of csv=p=0 "$output3_vf")

            if [ "$meta1" = "$meta2" ]; then
                echo "  [OK] Metadata Match: $meta1"
            else
                echo "  [NG] Metadata Mismatch!"
                echo "       File1: $meta1"
                echo "       File2: $meta2"
            fi
            if [ "$meta1" = "$meta3" ]; then
                echo "  [OK] Metadata Match (CUDA): $meta1"
            else
                echo "  [NG] Metadata Mismatch! (CUDA)"
                echo "       File1: $meta1"
                echo "       File2: $meta3"
            fi

            if [ "$meta1_vf" = "$meta2_vf" ]; then
                echo "  [OK] Metadata Match (vf): $meta1_vf"
            else
                echo "  [NG] Metadata Mismatch! (vf)"
                echo "       File1: $meta1_vf"
                echo "       File2: $meta2_vf"
            fi
            if [ "$meta1_vf" = "$meta3_vf" ]; then
                echo "  [OK] Metadata Match (CUDA vf): $meta1_vf"
            else
                echo "  [NG] Metadata Mismatch! (CUDA vf)"
                echo "       File1: $meta1_vf"
                echo "       File2: $meta3_vf"
            fi

            stats=$(ffmpeg -i "$output1" -i "$output2" -lavfi psnr -f null - 2>&1 | grep "PSNR y:")
            avg_psnr=$(echo "$stats" | sed -E 's/.*average:([0-9.]+).*/\1/')

            if [ -z "$avg_psnr" ]; then
                echo "  [!!] Error: Could not calculate PSNR"
            elif (( $(echo "$avg_psnr >= 38" | bc -l) )); then
                echo "  [OK] Quality Pass: PSNR $avg_psnr"
            else
                echo "  [FAIL] Quality Low: PSNR $avg_psnr"
            fi

            stats=$(ffmpeg -i "$output1_vf" -i "$output2_vf" -lavfi psnr -f null - 2>&1 | grep "PSNR y:")
            avg_psnr=$(echo "$stats" | sed -E 's/.*average:([0-9.]+).*/\1/')

            if [ -z "$avg_psnr" ]; then
                echo "  [!!] Error: Could not calculate PSNR"
            elif (( $(echo "$avg_psnr >= 38" | bc -l) )); then
                echo "  [OK] Quality Pass (vf): PSNR $avg_psnr"
            else
                echo "  [FAIL] Quality Low (vf): PSNR $avg_psnr"
            fi

            stats=$(ffmpeg -i "$output1" -i "$output3" -lavfi psnr -f null - 2>&1 | grep "PSNR y:")
            avg_psnr=$(echo "$stats" | sed -E 's/.*average:([0-9.]+).*/\1/')

            if [ -z "$avg_psnr" ]; then
                echo "  [!!] Error: Could not calculate PSNR"
            elif (( $(echo "$avg_psnr >= 38" | bc -l) )); then
                echo "  [OK] Quality Pass (CUDA): PSNR $avg_psnr"
            else
                echo "  [FAIL] Quality Low (CUDA): PSNR $avg_psnr"
            fi

            stats=$(ffmpeg -i "$output1_vf" -i "$output3_vf" -lavfi psnr -f null - 2>&1 | grep "PSNR y:")
            avg_psnr=$(echo "$stats" | sed -E 's/.*average:([0-9.]+).*/\1/')

            if [ -z "$avg_psnr" ]; then
                echo "  [!!] Error: Could not calculate PSNR"
            elif (( $(echo "$avg_psnr >= 38" | bc -l) )); then
                echo "  [OK] Quality Pass (CUDA vf): PSNR $avg_psnr"
            else
                echo "  [FAIL] Quality Low (CUDA vf): PSNR $avg_psnr"
            fi
        done
    done
done

# restore av
pip3 install av==${CURRENT_AV_VERSION}
