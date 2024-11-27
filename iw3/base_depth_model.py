from abc import ABCMeta, abstractmethod
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.models.data_parallel import DeviceSwitchInference
import os
from os import path
import pickle
import torch
from nunif.device import create_device
import numpy as np


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")


class EMAMinMax():
    def __init__(self, alpha=0.75):
        self.min_value = None
        self.max_value = None
        self.alpha = alpha

    def update(self, min_value, max_value):
        if self.min_value is None:
            self.min_value = float(min_value)
            self.max_value = float(max_value)
        else:
            self.min_value = self.alpha * self.min_value + (1. - self.alpha) * float(min_value)
            self.max_value = self.alpha * self.max_value + (1. - self.alpha) * float(max_value)

        # print(round(float(min_value), 3), round(float(max_value), 3), round(self.min_value, 3), round(self.max_value, 3))

        return self.min_value, self.max_value

    def __call__(self, min_value, max_value):
        return self.update(min_value, max_value)

    def reset(self):
        self.min_value = self.max_value = None

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.reset()


class BaseDepthModel(metaclass=ABCMeta):
    def __init__(self, model_type):
        self.device = None
        self.model = None
        self.model_type = model_type
        self.ema_minmax = EMAMinMax()
        self.ema_minmax_enabled = False

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

    @abstractmethod
    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0):
        pass

    def enable_ema_minmax(self, alpha):
        self.ema_minmax.set_alpha(alpha)
        self.ema_minmax_enabled = True

    def disable_ema_minmax(self):
        self.ema_minmax_enabled = False
        self.ema_minmax.reset()

    def reset_ema_minmax(self):
        self.ema_minmax.reset()

    def minmax_normalize_chw(self, depth):
        min_value = depth.amin()
        max_value = depth.amax()
        if self.ema_minmax_enabled:
            min_value, max_value = self.ema_minmax(min_value, max_value)

        # TODO: `1 - normalized_metric_depth` is wrong
        depth = 1.0 - ((depth - min_value) / (max_value - min_value))
        depth = depth.clamp(0, 1)
        depth = depth.nan_to_num()
        return depth

    def minmax_normalize(self, depth):
        if depth.ndim == 3:
            return self.minmax_normalize_chw(depth)
        else:
            assert depth.ndim == 4
            return torch.stack([self.minmax_normalize_chw(depth[i]) for i in range(depth.shape[0])]).contiguous()

    @staticmethod
    def normalized_depth_to_uint16_numpy(depth):
        uint16_max = 0xffff
        depth = uint16_max * depth
        depth = depth.to(torch.int16).numpy().astype(np.uint16)
        return depth

    def minmax_normalize_int16_numpy(depth):
        depth = self.minmax_normalize(depth)
        depth = self.depth_to_int16_numpy(depth)
        return depth

    @staticmethod
    def int16_depth_to_float32(depth):
        if depth.dtype != torch.float32:
            # 16bit image
            depth = torch.clamp(depth.to(torch.float32) / 0xffff, 0, 1)

        if depth.shape[0] != 1:
            # Maybe 24bpp
            # TODO: color depth support?
            depth = torch.mean(depth, dim=0, keepdim=True)

        # TODO: remove this. related to minmax_normalize_chw
        # invert
        depth = 1. - depth

        return depth


def _test():
    pass


if __name__ == "__main__":
    _test()
