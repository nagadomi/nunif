#!/bin/bash
PYTHON=python3
OUTPUT_DIR=./models/art_scan_psnr
ART_MODEL_DIR=./waifu2x/pretrained_models/swin_unet/art
DATA_DIR=./data/waifu2x

MAX_EPOCH=40
LR=0.00001
CYCLES=2
DEBUG=0
NUM_SAMPLES=25000
FORCE_SAVE_MODEL="--update-criterion all" # verification code does not support all photo noise, so without this, the best model may not be updated.
LOSS="--loss lbp5"
DA_OPTIONS="--deblur 0.05 --da-scale-p 0.5 --da-grayscale-p 0.01 --da-antialias-p 0.05"
LR_OPTIONS="--learning-rate-cycles ${CYCLES} --learning-rate ${LR} --max-epoch ${MAX_EPOCH}"
# use `--style photo` for photo noise
OPTIONS="--arch waifu2x.swin_unet_4x --style photo ${LOSS} --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --size 64 --batch-size 16 --num-samples ${NUM_SAMPLES} ${DA_OPTIONS} ${LR_OPTIONS} --disable-backup ${FORCE_SAVE_MODEL}"


DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 3 ${OPTIONS} --checkpoint-file ${ART_MODEL_DIR}/noise3_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 ${OPTIONS} --checkpoint-file ${ART_MODEL_DIR}/noise2_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 ${OPTIONS} --checkpoint-file ${ART_MODEL_DIR}/noise1_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 ${OPTIONS} --checkpoint-file ${ART_MODEL_DIR}/noise0_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x ${OPTIONS} --checkpoint-file ${ART_MODEL_DIR}/scale4x.pth

_=<<__COMMENT_OUT
# finetune again
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 3 ${OPTIONS} --resume --reset-state --learning-rate-cycles 1 --max-epoch 10 
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 ${OPTIONS} --resume --reset-state --learning-rate-cycles 1 --max-epoch 10
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 ${OPTIONS} --resume --reset-state --learning-rate-cycles 1 --max-epoch 10
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 ${OPTIONS} --resume --reset-state --learning-rate-cycles 1 --max-epoch 10
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x ${OPTIONS} --resume --reset-state --learning-rate-cycles 1 --max-epoch 10
__COMMENT_OUT
