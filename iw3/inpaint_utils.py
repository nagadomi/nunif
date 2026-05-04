import sys
from os import path

import yaml

from nunif.models import load_model
from nunif.utils.home_dir import ensure_home_dir
from nunif.utils.ui import TorchHubDir

from .hub_dir import HUB_MODEL_DIR


def pth_url(filename):
    return "https://github.com/nagadomi/nunif/releases/download/0.0.0/" + filename


MASK_MLBW_L2_D1_URL = pth_url("iw3_mask_mlbw_l2_d1_20250903.pth")
INPAINT_CONFIG_FILE = path.join(ensure_home_dir("iw3"), "inpaint_models.yml")
INPAINT_MODEL_DEFAULT = "light_inpaint_v1"


def _resolve_path(path_or_url):
    if not path_or_url:
        return path_or_url

    if path_or_url.lower().startswith(("http://", "https://")):
        return path_or_url

    if path.isabs(path_or_url):
        return path_or_url

    # Relative Path
    repository_root = ensure_home_dir(None)
    return path.normpath(path.join(repository_root, path_or_url))


def _load_inpaint_model_list():
    inpaint_models = {
        INPAINT_MODEL_DEFAULT: {
            "video": pth_url("iw3_light_video_inpaint_v1_20250919.pth"),
            "image": pth_url("iw3_light_inpaint_v1_20250919.pth"),
        }
    }
    if path.exists(INPAINT_CONFIG_FILE):
        with open(INPAINT_CONFIG_FILE, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"{INPAINT_CONFIG_FILE}: Error: {e}", file=sys.stderr)
                config = None

        if isinstance(config, dict):
            for name, value in config.items():
                if not isinstance(value, dict):
                    continue
                if name in inpaint_models:
                    continue

                inpaint_models[name] = {
                    "video": _resolve_path(value.get("video")),
                    "image": _resolve_path(value.get("image")),
                }

                # Use the default model when the video or image path is not defined
                if inpaint_models[name]["video"] is None:
                    inpaint_models[name]["video"] = inpaint_models[INPAINT_MODEL_DEFAULT]["video"]
                if inpaint_models[name]["image"] is None:
                    inpaint_models[name]["image"] = inpaint_models[INPAINT_MODEL_DEFAULT]["image"]

    return inpaint_models


INPAINT_MODELS = _load_inpaint_model_list()


def load_image_inpaint_model(name, device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        if name is None:
            name = INPAINT_MODEL_DEFAULT
        if name not in INPAINT_MODELS:
            raise ValueError(f"inpaint model `{name}` is not defined")
        model, _ = load_model(INPAINT_MODELS[name]["image"], device_ids=[device_id], weights_only=True)
        return model.eval()


def load_video_inpaint_model(name, device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        if name is None:
            name = INPAINT_MODEL_DEFAULT
        if name not in INPAINT_MODELS:
            raise ValueError(f"inpaint model `{name}` is not defined")
        model, _ = load_model(INPAINT_MODELS[name]["video"], device_ids=[device_id], weights_only=True)
        return model.eval()


def load_mask_mlbw(device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        model, _ = load_model(MASK_MLBW_L2_D1_URL, device_ids=[device_id], weights_only=True)
        model.delta_output = True
        return model.eval()


class CompileContext:
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False
