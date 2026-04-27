#!/bin/bash -e

echo "**** ${0}"

TEST_IMAGE=tests/data/smoke/sd.png
TEST_VIDEO=tests/data/smoke/sd.mkv
TEST_DIR=tests/data/smoke/
OUTPUT_DIR=tests/data/smoke/waifu2x
MODEL_DIR=waifu2x/pretrained_models/upconv_7/art

set -x

python -m waifu2x.download_models
python -m waifu2x.cli -y -i ${TEST_IMAGE} -o ${OUTPUT_DIR} --noise-level 0 -m noise_scale --model-dir ${MODEL_DIR}
python -m waifu2x.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --noise-level 1 -m noise --model-dir ${MODEL_DIR}
python -m waifu2x.cli -y -i ${TEST_DIR} -o ${OUTPUT_DIR} -m scale --model-dir ${MODEL_DIR}
