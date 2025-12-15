FROM nvidia/cuda:12.8.1-base-ubuntu24.04
LABEL maintainer="nagadomi <https://github.com/nagadomi>"

# install deps
RUN apt-get update -y && apt-get install -y git-core python3-dev python3-pip libmagickwand-dev libraqm-dev

# install
RUN git clone https://github.com/nagadomi/nunif.git /root/nunif && \
    cd /root/nunif && \
    pip3 install --break-system-packages -r requirements-torch.txt && \
    pip3 install --break-system-packages -r requirements.txt && \
    pip3 install --break-system-packages -r requirements-dev.txt && \
    python3 -m waifu2x.download_models && \
    python3 -m waifu2x.web.webgen.gen && \
    python3 -m iw3.download_models

WORKDIR /root/nunif
