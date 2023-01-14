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

if [ $# -eq 2 ]; then
    gpu=$1
    port=$2
fi

source .venv/bin/activate
# TODO: log rotate
mkdir -p ./tmp/logs
python3 -u -m waifu2x.web --image-lib pil --port ${port} --gpu ${gpu} ${debug_log} ${recaptcha} >> ./tmp/logs/waifu2x_${port}.log 2>&1
