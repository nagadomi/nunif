#!/bin/bash
# Train photo 4x models
# photo model is trained only for 4x. 2x and 1x models are converted from the 4x model (bicubic downscaling).

# 1. Place the pretrained scale4x.pth in OUTPUT_DIR first. 

PYTHON=python3
OUTPUT_DIR=./models/photo_psnr
DATA_DIR=./data/photo
MAX_EPOCH=40
LR=0.00003
CYCLES=2
NUM_SAMPLES=25000
FORCE_SAVE_MODEL="--update-criterion all" # use "--update-criterion all" if needed. validation does not support all photo noise so it may no longer update the best models.
LOSS="--loss lbp5" # lbp5 for 4x, lbp5m for unif4x (includes downscaled 2x training)
DA_OPTIONS="--deblur 0.05 --da-scale-p 0.75 --da-unsharpmask-p 0.2 --da-grayscale-p 0.01 --da-color-p 0.5"
LR_OPTIONS="--learning-rate-cycles ${CYCLES} --learning-rate ${LR} --max-epoch ${MAX_EPOCH}"
OPTIONS="--arch waifu2x.swin_unet_4x --style photo ${LOSS} --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --size 64 --batch-size 16 --num-samples ${NUM_SAMPLES} ${DA_OPTIONS} ${LR_OPTIONS} --disable-backup ${FORCE_SAVE_MODEL} --hard-example none "
DEBUG=0

# 4x, 0 to 3
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise0_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise1_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 3 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise2_scale4x.pth

# 3 to 0, 4x
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise3_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise2_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise1_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise0_scale4x.pth

_=<<__COMMENT_OUT
# finetune only
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method scale4x ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 0 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise0_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 1 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise1_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 2 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise2_scale4x.pth
DEBUG=${DEBUG} ${PYTHON} train.py waifu2x --method noise_scale4x --noise-level 3 ${OPTIONS} --checkpoint-file ${OUTPUT_DIR}/noise3_scale4x.pth
__COMMENT_OUT
