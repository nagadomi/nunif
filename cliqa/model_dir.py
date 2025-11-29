from os import path
from nunif.utils.home_dir import ensure_home_dir

MODEL_DIR = path.join(ensure_home_dir("cliqa"), "pretrained_models")
