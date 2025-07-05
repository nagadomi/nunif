from abc import ABCMeta, abstractmethod
import contextlib
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.models.utils import compile_model
import os
from os import path
import pickle
import torch
from nunif.device import create_device
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import functional as TF
from . depth_scaler import EMAMinMaxScaler


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")


class _CompileContext():
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False


class BaseDepthModel(metaclass=ABCMeta):
    def __init__(self, model_type):
        self.device = None
        self.model = None
        self.model_backup = None  # for compile
        self.model_type = model_type
        self.scaler = EMAMinMaxScaler(decay=0, buffer_size=1)

    def compile_context(self, enabled=True):
        if enabled:
            return _CompileContext(self)
        else:
            return contextlib.nullcontext()

    @classmethod
    @abstractmethod
    def get_name():
        pass

    def loaded(self):
        return self.model is not None

    @classmethod
    @abstractmethod
    def supported(cls, model_type):
        pass

    @classmethod
    @abstractmethod
    def has_checkpoint_file(cls, model_type):
        pass

    @classmethod
    @abstractmethod
    def get_model_path(cls, model_type):
        pass

    @abstractmethod
    def is_metric(self):
        pass

    def is_image_supported(self):
        return True

    def is_video_supported(self):
        return True

    @staticmethod
    def force_update_hub(github, model):
        with TorchHubDir(HUB_MODEL_DIR):
            torch.hub.help(github, model, force_reload=True, trust_repo=True)

    @classmethod
    @abstractmethod
    def force_update(cls):
        pass

    @classmethod
    @abstractmethod
    def multi_gpu_supported(cls, name):
        pass

    @abstractmethod
    def load_model(self, model_type, resolution, device):
        pass

    def load(self, gpu=0, resolution=None):
        self.device = create_device(gpu)

        with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
            try:
                self.model = self.load_model(self.model_type, resolution=resolution, device=self.device)
            except (RuntimeError, pickle.PickleError) as e:
                if isinstance(e, RuntimeError):
                    do_handle = "PytorchStreamReader" in repr(e)
                else:
                    do_handle = True
                if do_handle:
                    try:
                        # delete corrupted file
                        os.unlink(self.get_model_path(self.model_type))
                    except:  # noqa
                        pass
                    raise RuntimeError(
                        f"File `{self.get_model_path(self.model_type)}` is corrupted. "
                        "This error may occur when the network is unstable or the disk is full. "
                        "Try again."
                    )
                else:
                    raise

        self.model = self.model.to(self.device).eval()

        if (isinstance(gpu, (list, tuple)) and len(gpu) > 1):
            if self.multi_gpu_supported(self.model_type):
                self.model = DeviceSwitchInference(self.model, device_ids=gpu)
            else:
                raise ValueError(f"{self.model_type} does not support Multi-GPU")

        return self

    def get_model(self):
        return self.model

    def compile(self):
        if self.model_backup is None and not isinstance(self.model, DeviceSwitchInference):
            self.model_backup = self.model
            self.model = compile_model(self.model)

    def clear_compiled_model(self):
        if self.model_backup is not None:
            self.model = self.model_backup
            self.model_backup = None

    @abstractmethod
    def infer(self, x, *kwargs):
        pass

    def enable_ema(self, decay, buffer_size=None):
        self.scaler.reset(decay=decay, buffer_size=buffer_size)

    def get_ema_state(self):
        return self.scaler.decay, self.scaler.buffer_size

    def disable_ema(self):
        self.scaler.reset(decay=0, buffer_size=1)

    def reset_ema(self, decay=None, buffer_size=None):
        self.scaler.reset(decay=decay, buffer_size=buffer_size)

    def reset_state(self):
        pass

    def reset(self):
        self.reset_ema()
        self.reset_state()

    def get_ema_buffer_size(self):
        return self.scaler.buffer_size

    def minmax_normalize_chw(self, depth, return_minmax=False):
        return self.scaler(depth, return_minmax=return_minmax)

    def flush_minmax_normalize(self, return_minmax=False):
        return self.scaler.flush(return_minmax=return_minmax)

    def minmax_normalize(self, depth, reset_ema=None):
        assert depth.ndim == 4
        reset_ema = [False] * depth.shape[0] if reset_ema is None else reset_ema
        assert len(reset_ema) == depth.shape[0]
        normalized_depths = []
        for i in range(depth.shape[0]):
            normalized_depth = self.minmax_normalize_chw(depth[i])
            if normalized_depth is not None:
                normalized_depths.append(normalized_depth)
            if reset_ema[i]:
                normalized_depths += self.flush_minmax_normalize()
                self.reset_ema()
        return normalized_depths

    @staticmethod
    def save_normalized_depth(depth, file_path, png_info={}, min_depth_value=None, max_depth_value=None):
        if min_depth_value is not None:
            png_info.update(iw3_min_depth_value=float(min_depth_value))
        if max_depth_value is not None:
            png_info.update(iw3_max_depth_value=float(max_depth_value))

        depth = torch.clamp(depth, 0, 1)
        depth_int = (0xffff * depth).to(torch.uint16).squeeze(0).cpu().numpy()
        metadata = PngInfo()
        for k, v in png_info.items():
            metadata.add_text(k, str(v))

        im = Image.fromarray(depth_int)
        im.save(file_path, pnginfo=metadata)

    @staticmethod
    def load_depth(file_path):
        # TODO: test iw3_max_depth_value
        with Image.open(file_path) as im:
            if "iw3_min_depth_value" in im.text and "iw3_max_depth_value" in im.text:
                try:
                    min_depth_value = float(im.text["iw3_min_depth_value"])
                    max_depth_value = float(im.text["iw3_max_depth_value"])
                except (ValueError, TypeError):
                    min_depth_value = max_depth_value = None
            else:
                min_depth_value = max_depth_value = None

            depth = TF.pil_to_tensor(im)
            if depth.dtype != torch.float32:
                depth = torch.clamp(depth.to(torch.float32) / 0xffff, 0, 1)
            if depth.shape[0] != 1:
                depth = torch.mean(depth, dim=0, keepdim=True)

            if min_depth_value is not None and max_depth_value is not None:
                depth = depth * (max_depth_value - min_depth_value) + min_depth_value

            metadata = {}
            metadata.update(im.text)
            metadata.update({"filename": file_path})

            return depth, metadata


def _test():
    pass


if __name__ == "__main__":
    _test()
