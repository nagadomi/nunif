#!/bin/bash
PYTHON=python3
OUTPUT_DIR=./models/art_scan_gan
PSNR_MODEL_DIR=./models/art_scan_psnr
DATA_DIR=./data/waifu2x

MAX_EPOCH_S=30
NUM_SAMPLES=25000
DEBUG=0
LR_S=3e-07
DISCRIMINATOR=u3c
LOSS=rgb_l1lbp5
BATCH_SIZE=16
DISCRIMINATOR_WEIGHT=1
LR_OPTIONS=" --scheduler step --learning-rate-decay 0.3 --learning-rate-decay-step 3 10 20 --adam-beta1 0.75 --discriminator-weight ${DISCRIMINATOR_WEIGHT} --generator-start-criteria 0.999"
DA_OPTIONS=" --deblur 0.05 --da-scale-p 0.5 --da-grayscale-p 0.01 --da-antialias-p 0.05 "
# use `--style photo` for photo noise
OPTIONS=" --arch waifu2x.swin_unet_4x --style photo  --loss ${LOSS} --discriminator ${DISCRIMINATOR} --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --num-samples ${NUM_SAMPLES} ${LR_OPTIONS} ${DA_OPTIONS} --update-criterion all --disable-backup --hard-example none "
OPTIONS_S=" ${OPTIONS} --size 64  --batch-size ${BATCH_SIZE} --learning-rate ${LR_S} --max-epoch ${MAX_EPOCH_S} "

DEBUG=${DEBUG} ${PYTHON} train.py waifu2x ${OPTIONS_S} --method noise_scale4x --noise-level 3 --checkpoint-file ${PSNR_MODEL_DIR}/noise3_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x ${OPTIONS_S} --method noise_scale4x --noise-level 2 --checkpoint-file ${PSNR_MODEL_DIR}/noise2_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x ${OPTIONS_S} --method noise_scale4x --noise-level 1 --checkpoint-file ${PSNR_MODEL_DIR}/noise1_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x ${OPTIONS_S} --method noise_scale4x --noise-level 0 --checkpoint-file ${PSNR_MODEL_DIR}/noise0_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x ${OPTIONS_S} --method scale4x --checkpoint-file ${PSNR_MODEL_DIR}/scale4x.pth
