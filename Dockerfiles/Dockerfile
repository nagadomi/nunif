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
