#!/bin/bash -e

OUTPUT_DIR=./models/swin_unet_v3_1x/
DATA_DIR=./data/waifu2x/
ADDITIONAL_DATA_DIR=./data/waifu2x/gen_screenmain
#ADDITIONAL_DATA_DIR=
# random
SEED=-1

ARCH="waifu2x.swin_unet_v3_1x"
DA_OPTION="--da-grayscale-p 0.02 --da-scale-p 0.25 --da-mixup-p 0.005 --da-cutmix-p 0.005 --da-color-p 0.05 "
OPTIONS="--arch ${ARCH} --optimizer adamw_fused --scheduler cosine_wd --update-criterion loss --seed ${SEED} --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --drop-last  --eval-step 2 --num-workers 8"
if [ ! -z ${ADDITIONAL_DATA_DIR} ]; then
    OPTIONS="${OPTIONS} --additional-data-dir ${ADDITIONAL_DATA_DIR} --additional-data-dir-p 0.02 --hard-example none"
fi

OPTIONS="${OPTIONS} ${DA_OPTION} --ignore-nan "
OVERRIDE=""

# first train --noise-level 1
LOSS="--loss dctirm"

echo "***** ${ARCH} *****"

# *** noise 1

# train
STEP_OPTION="--size 64 --batch-size 16 --num-samples 50000 --max-epoch 200  --learning-rate-cycles 5 --learning-rate 0.0002 --warmup-epoch 1"
DEBUG=1 python train.py waifu2x --method noise --noise-level 1 ${OPTIONS} ${STEP_OPTION} ${LOSS}
cp ${OUTPUT_DIR}/noise1.pth ${OUTPUT_DIR}/noise1.base.0.pth

STEP_OPTION="--size 80 --batch-size 16 --num-samples 50000 --max-epoch 200 --learning-rate-cycles 5 --learning-rate 0.0001"
DEBUG=1 python train.py waifu2x --method noise --noise-level 1 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise1.base.0.pth
cp ${OUTPUT_DIR}/noise1.pth ${OUTPUT_DIR}/noise1.base.1.pth

STEP_OPTION="--size 128 --batch-size 16 --backward-step 4 --num-samples 20000 --max-epoch 200 --learning-rate-cycles 5 --learning-rate 0.0001"
DEBUG=1 python train.py waifu2x --method noise --noise-level 1 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise1.base.1.pth
cp ${OUTPUT_DIR}/noise1.pth ${OUTPUT_DIR}/noise1.base.2.pth


# *** noise 2 3
STEP_OPTION="--size 80 --batch-size 16 --num-samples 50000 --max-epoch 200  --learning-rate-cycles 5 --learning-rate 0.0001 --warmup-epoch 1"
DEBUG=1 python train.py waifu2x --method noise --noise-level 2 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise1.base.2.pth
cp ${OUTPUT_DIR}/noise2.pth ${OUTPUT_DIR}/noise2.0.pth


STEP_OPTION="--size 80 --batch-size 16 --num-samples 50000 --max-epoch 200  --learning-rate-cycles 5 --learning-rate 0.0001 --warmup-epoch 1"
DEBUG=1 python train.py waifu2x --method noise --noise-level 3 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise2.0.pth
cp ${OUTPUT_DIR}/noise3.pth ${OUTPUT_DIR}/noise3.0.pth

# *** noise 1 0 -1

STEP_OPTION="--size 80 --batch-size 16 --num-samples 50000 --max-epoch 200  --learning-rate-cycles 5 --learning-rate 0.0001 --warmup-epoch 1"
DEBUG=1 python train.py waifu2x --method noise --noise-level 1 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise3.0.pth
cp ${OUTPUT_DIR}/noise1.pth ${OUTPUT_DIR}/noise1.0.pth

STEP_OPTION="--size 80 --batch-size 16 --num-samples 50000 --max-epoch 200  --learning-rate-cycles 5 --learning-rate 0.0001 --warmup-epoch 1"
DEBUG=1 python train.py waifu2x --method noise --noise-level 0 ${OPTIONS} ${STEP_OPTION} ${LOSS} --checkpoint-file ${OUTPUT_DIR}/noise1.0.pth
cp ${OUTPUT_DIR}/noise0.pth ${OUTPUT_DIR}/noise0.0.pth
