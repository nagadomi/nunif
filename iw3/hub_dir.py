import os
from os import path

from nunif.utils.home_dir import ensure_home_dir

HUB_MODEL_DIR = path.join(ensure_home_dir("iw3"), "pretrained_models", "hub")
os.makedirs(path.join(HUB_MODEL_DIR, "checkpoints"), exist_ok=True)
