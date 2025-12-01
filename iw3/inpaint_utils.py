import sys
from os import path
import yaml
import torch
from nunif.utils.ui import TorchHubDir
from nunif.utils.home_dir import ensure_home_dir
from nunif.models import load_model
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

                inpaint_models[name] = {"video": _resolve_path(value.get("video")), "image": _resolve_path(value.get("image"))}

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
            raise ValueError(f"inpaint_name={name} is not defined")
        model, _ = load_model(INPAINT_MODELS[name]["image"], device_ids=[device_id], weights_only=True)
        return model.eval()


def load_video_inpaint_model(name, device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        if name is None:
            name = INPAINT_MODEL_DEFAULT
        if name not in INPAINT_MODELS:
            raise ValueError(f"inpaint_name={name} is not defined")
        model, _ = load_model(INPAINT_MODELS[name]["video"], device_ids=[device_id], weights_only=True)
        return model.eval()


def load_mask_mlbw(device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        model, _ = load_model(MASK_MLBW_L2_D1_URL, device_ids=[device_id], weights_only=True)
        model.delta_output = True
        return model.eval()


class FrameQueue():
    def __init__(
            self,
            synthetic_view, seq, height, width, dtype, device,
            mask_height=None, mask_width=None
    ):
        if mask_width is None:
            mask_width = width
        if mask_height is None:
            mask_height = height

        self.left_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        self.right_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        if synthetic_view == "both":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
        elif synthetic_view == "right":
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.left_mask = None
        elif synthetic_view == "left":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = None

        self.synthetic_view = synthetic_view
        self.index = 0
        self.max_index = seq

    def full(self):
        return self.index == self.max_index

    def empty(self):
        return self.index == 0

    def add(self, left_eye, right_eye, left_mask=None, right_mask=None):
        self.left_eye[self.index] = left_eye
        self.right_eye[self.index] = right_eye
        if left_mask is not None:
            self.left_mask[self.index] = left_mask
        if right_mask is not None:
            self.right_mask[self.index] = right_mask

        self.index += 1

    def fill(self):
        if self.full():
            return 0

        pad = 0
        i = self.index - 1
        if self.synthetic_view == "both":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "right":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "left":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone())
        while not self.full():
            pad += 1
            self.add(**frame)

        return pad

    def remove(self, n):
        if n > 0 and n < self.max_index:
            for i in range(n):
                self.left_eye[i] = self.left_eye[i + n]
                self.right_eye[i] = self.right_eye[i + n]
                if self.right_mask is not None:
                    self.right_mask[i] = self.right_mask[i + n]
                if self.left_mask is not None:
                    self.left_mask[i] = self.left_mask[i + n]

        self.index -= n
        assert self.index >= 0

    def get(self):
        if self.synthetic_view == "both":
            return self.left_eye, self.right_eye, self.left_mask, self.right_mask
        elif self.synthetic_view == "left":
            return self.left_eye, self.right_eye, self.left_mask
        elif self.synthetic_view == "right":
            return self.left_eye, self.right_eye, self.right_mask

    def clear(self):
        self.index = 0


class CompileContext():
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False
