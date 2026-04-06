#!/bin/bash -e

echo "**** ${0}"

TEST_VIDEO=tests/data/smoke/sd.mkv
TEST_VIDEO_HDR=tests/data/smoke/hdr.mkv
TEST_DIR=tests/data/smoke/
OUTPUT_DIR=tests/data/smoke_hwaccel/iw3
TEST_YAML=${OUTPUT_DIR}/sd/iw3_export.yml

H264_ENC=h264_nvenc
H265_ENC=hevc_nvenc

HWACCEL="--hwaccel cuda"
#HWACCEL="--hwaccel cuda_hwdownload"
#HWACCEL="--hwaccel cuda_hwdownload --disable-software-fallback"
#HWACCEL="--hwaccel vaapi --disable-software-fallback"

set -x

# base
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --video-codec ${H264_ENC} ${HWACCEL}
python -m iw3.cli -y -i ${TEST_DIR} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --resume --video-codec ${H265_ENC} ${HWACCEL}

# EMA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10 --video-codec ${H264_ENC} ${HWACCEL}

# Batch
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream --video-codec ${H264_ENC} ${HWACCEL}

# Low VRAM
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --low-vram --video-codec ${H264_ENC} ${HWACCEL}

# VDA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --disable-scene-cache --video-codec ${H264_ENC} ${HWACCEL}

# VDA Stream
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --video-codec ${H264_ENC} ${HWACCEL}

# HDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace auto --video-codec libx265 --pix-fmt yuv420p10le --video-codec ${H265_ENC} ${HWACCEL}
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10 --scene-detect --pix-fmt yuv420p10le --video-codec ${H265_ENC} ${HWACCEL}

# HDR2SDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace bt709-tv --video-codec libx265 --pix-fmt yuv420p --video-codec ${H265_ENC} ${HWACCEL}

# inpaint
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_inpaint --video-codec ${H264_ENC} ${HWACCEL}
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10 --scene-detect --video-codec ${H265_ENC} ${HWACCEL}

# export
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --export-disparity ${HWACCEL}
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --export --export-depth-only --export-depth-fit --ema-normalize --ema-buffer 10 --scene-detect ${HWACCEL}

# import
python -m iw3.cli -y -i ${TEST_YAML} -o ${OUTPUT_DIR} --video-codec ${H264_ENC}

# vf
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --vf "scale=-2:320,crop=256:256" ${HWACCEL}
