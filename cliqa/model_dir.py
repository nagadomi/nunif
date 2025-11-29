from os import path
from nunif.utils.home_dir import ensure_home_dir

MODEL_DIR = path.join(ensure_home_dir("cliqa", path.dirname(__file__)), "pretrained_models")
