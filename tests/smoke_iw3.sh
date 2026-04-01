#!/bin/bash -e

echo "**** ${0}"

TEST_IMAGE=tests/data/smoke/sd.png
TEST_VIDEO=tests/data/smoke/sd.mkv
TEST_VIDEO_HDR=tests/data/smoke/hdr.mkv
TEST_DIR=tests/data/smoke/
OUTPUT_DIR=tests/data/smoke/iw3

set -x

# base
python -m iw3.cli -y -i ${TEST_IMAGE} -o ${OUTPUT_DIR} --depth-model Any_S --metadata
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata
python -m iw3.cli -y -i ${TEST_DIR} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --resume

# EMA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10

# Batch
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream

# Low VRAM
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --low-vram

# VDA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --disable-scene-cache

# VDA Stream
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect

# HDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace auto --video-codec libx265 --pix-fmt yuv420p10le
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10 --scene-detect  --video-codec libx265 --pix-fmt yuv420p10le

# HDR2SDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace bt709-tv --video-codec libx265 --pix-fmt yuv420p

# inpaint
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_inpaint
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10 --scene-detect

# export
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --export-disparity --export-depth-only --export-depth-fit
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --export --ema-normalize --ema-buffer 10 --scene-detect
