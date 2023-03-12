FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
MAINTAINER nagadomi <nagadomi@gmail.com>

# install deps
RUN apt-get update -y && apt-get install -y git-core python3-pip libmagickwand-dev libraqm-dev

# install
RUN git clone https://github.com/nagadomi/nunif.git /root/nunif && \
    cd /root/nunif && \
    pip3 install torch torchvision torchaudio torchtext && \
    pip3 install -r requirements.txt && \
    python3 -m waifu2x.download_models && \
    python3 -m waifu2x.web.webgen.gen

WORKDIR /root/nunif

# 1. Build
# 
# docker build -t nunif .
# 
# 2. For CUDA
# docker run --gpus all -p 8812:8812 --rm nunif python3 -m waifu2x.web --port 8812 --bind-addr 0.0.0.0 --max-pixels 16777216 --max-body-size 100
# 
# 2. For CPU
# docker run -p 8812:8812 --rm nunif python3 -m waifu2x.web --port 8812 --bind-addr 0.0.0.0 --max-pixels 16777216 --max-body-size 100 --gpu -1
#
# Open http://localhost:8812/ 
