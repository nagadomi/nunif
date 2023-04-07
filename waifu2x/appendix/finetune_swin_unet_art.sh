#!/bin/bash

# finetune command example for swin_unet_*x
# see also docs/training.md

# cp -r waifu2x/pretrained_models/swin_unet/art ./models/swin_unet_1
# ./waifu2x/appendix/finetune_swin_unet_all.sh


PYTHON=python3
OUTPUT_DIR=./models/swin_unet_1
DATA_DIR=./data/waifu2x
CYCLES=1
MAX_EPOCH=30
LR=0.00005

# 2x
DEBUG=1 $PYTHON train.py waifu2x --method scale --arch waifu2x.swin_unet_2x --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --warmup-epoch 0 --loss lbp --size 64 --batch-size 16 --optimizer adamw  --learning-rate ${LR} --learning-rate-cycles ${CYCLES} --max-epoch ${MAX_EPOCH} --deblur 0.025 --checkpoint-file ${OUTPUT_DIR}/scale2x.pth

for ((i = 0; i <= 3; ++i)); do
    DEBUG=1 $PYTHON train.py waifu2x --method noise_scale --noise-level ${i} --arch waifu2x.swin_unet_2x --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --warmup-epoch 0 --loss lbp --size 64 --batch-size 16 --optimizer adamw  --learning-rate ${LR} --learning-rate-cycles ${CYCLES} --max-epoch ${MAX_EPOCH} --deblur 0.025 --checkpoint-file ${OUTPUT_DIR}/noise${i}_scale2x.pth
done

# 4x

DEBUG=1 $PYTHON train.py waifu2x --method scale4x --arch waifu2x.swin_unet_4x --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --warmup-epoch 0 --loss lbp5 --size 64 --batch-size 16 --optimizer adamw  --learning-rate ${LR} --learning-rate-cycles ${CYCLES} --max-epoch ${MAX_EPOCH} --deblur 0.05 --checkpoint-file ${OUTPUT_DIR}/scale4x.pth

for ((i = 0; i <= 3; ++i)); do
    DEBUG=1 $PYTHON train.py waifu2x --method noise_scale4x --noise-level ${i} --arch waifu2x.swin_unet_4x --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --warmup-epoch 0 --loss lbp5 --size 64 --batch-size 16 --optimizer adamw  --learning-rate ${LR} --learning-rate-cycles ${CYCLES} --max-epoch ${MAX_EPOCH} --deblur 0.05 --checkpoint-file ${OUTPUT_DIR}/noise${i}_scale4x.pth
done

# 1x (denoise)
for ((i = 0; i <= 3; ++i)); do
    DEBUG=1 $PYTHON train.py waifu2x --method noise --noise-level ${i} --arch waifu2x.swin_unet_1x --data-dir ${DATA_DIR} --model-dir ${OUTPUT_DIR} --warmup-epoch 0 --loss lbp --size 64 --batch-size 16 --optimizer adamw  --learning-rate ${LR} --learning-rate-cycles ${CYCLES} --max-epoch ${MAX_EPOCH} --checkpoint-file ${OUTPUT_DIR}/noise${i}.pth
done


# comment out example
_=<<__COMMENT_OUT

this is block comment

__COMMENT_OUT
