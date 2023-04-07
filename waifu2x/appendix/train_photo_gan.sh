#!/bin/bash

# this requires 8GB VRAM (I used 1x RTX3070 Ti)
# If out of memory errors occur, decreasing --batch-size to 14,12,8,...
PYTHON=python3
OUTPUT_DIR=./models/photo_gan
PSNR_MODEL_DIR=./models/photo_psnr
DATA_DIR=./data/photo

MAX_EPOCH_S=40
MAX_EPOCH_L=10
NUM_SAMPLES=25000
DEBUG=0
LR_S=0.00003
LR_L=0.00001
LR_OPTIONS=" --scheduler step --learning-rate-decay 0.3 --learning-rate-decay-step 10 --adam-beta1 0.75"
DA_OPTIONS=" --deblur 0.05 --da-scale-p 0.75 --da-unsharpmask-p 0.2 --da-grayscale-p 0.01 --da-color-p 0.5 "
OPTIONS=" --arch waifu2x.swin_unet_4x --style photo  --loss l1lpips --discriminator l3v1c --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --num-samples ${NUM_SAMPLES} ${LR_OPTIONS} ${DA_OPTIONS} --update-criterion all --disable-backup --hard-example none "
OPTIONS_S=" ${OPTIONS} --size 64  --batch-size 12 --learning-rate ${LR_S} --max-epoch ${MAX_EPOCH_S} "
OPTIONS_L=" ${OPTIONS} --size 112 --batch-size 4  --learning-rate ${LR_L} --max-epoch ${MAX_EPOCH_L} " # large input and slow setting

DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x --checkpoint-file ${PSNR_MODEL_DIR}/scale4x.pth ${OPTIONS_S}
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 --checkpoint-file ${PSNR_MODEL_DIR}/noise0_scale4x.pth ${OPTIONS_S}
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 --checkpoint-file ${PSNR_MODEL_DIR}/noise1_scale4x.pth ${OPTIONS_S}
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 --checkpoint-file ${PSNR_MODEL_DIR}/noise2_scale4x.pth ${OPTIONS_S}
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 3 --checkpoint-file ${PSNR_MODEL_DIR}/noise3_scale4x.pth ${OPTIONS_S}
