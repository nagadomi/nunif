#!/bin/bash

# waifu2x daemon script
# run-web.sh <gpu id> <port>
# log_dir: tmp/logs/waifu2x_{port}.log

gpu=0
port=8812
#debug_log=""
debug_log="--debug"
recaptcha=""
#recaptcha="--enable-recaptcha --config ./waifu2x/web/config.ini"

# Specify this option if you want to full compile the model.
# The first compile takes several minutes, but the second and subsequent compiles take about 20 seconds with pytorch >= 2.5.1.
# example: python3 -u -m waifu2x.web ${compile_flags}
compile_flags=" --compile --warmup --batch-size 1"

if [ $# -eq 2 ]; then
    gpu=$1
    port=$2
fi

source .venv/bin/activate
# TODO: log rotate
mkdir -p ./tmp/logs
python3 -u -m waifu2x.web --image-lib pil --port ${port} --gpu ${gpu} ${debug_log} ${recaptcha} >> ./tmp/logs/waifu2x_${port}.log 2>&1
