from os import path
from nunif.utils.home_dir import ensure_home_dir

HUB_MODEL_DIR = path.join(ensure_home_dir("iw3", path.dirname(__file__)), "pretrained_models", "hub")
