from abc import ABCMeta, abstractmethod
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.models.data_parallel import DeviceSwitchInference
import os
from os import path
import pickle
import torch
from nunif.device import create_device
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import functional as TF


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
        self.model_backup = None  # for compile
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
            self.model = torch.compile(self.model)

    def clear_compile(self):
        if self.model_backup is not None:
            self.model = self.model_backup
            self.model_backup = None

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
        return depth, min_value, max_value

    def minmax_normalize(self, depth):
        if depth.ndim == 3:
            return self.minmax_normalize_chw(depth)[0]
        else:
            assert depth.ndim == 4
            return torch.stack([self.minmax_normalize_chw(depth[i])[0] for i in range(depth.shape[0])]).contiguous()

    def save_depth(self, depth, file_path, png_info={}, normalize=True):
        # not batch
        assert depth.ndim in {3, 2}
        if normalize:
            depth, min_depth_value, max_depth_value = self.minmax_normalize_chw(depth)
            png_info.update(iw3_min_depth_value=min_depth_value.item(), iw3_max_depth_value=max_depth_value.item())
        else:
            min_depth_value = depth.amin()
            max_depth_value = depth.amax()
            if not (0 - 1e-6 <= min_depth_value and max_depth_value <= 1.0 + 1e-6):
                raise ValueError("depth is not normalized and normalize=False."
                                 f"min={min_depth_value}, max={max_depth_value}")

        depth_int = (0xffff * depth).to(torch.uint16).squeeze(0).cpu().numpy()
        metadata = PngInfo()
        for k, v in png_info.items():
            metadata.add_text(k, str(v))

        im = Image.fromarray(depth_int)
        im.save(file_path, pnginfo=metadata)

    @staticmethod
    def load_depth(file_path):
        with Image.open(file_path) as im:
            if "iw3_min_depth_value" in im.text and "iw3_max_depth_value" in im.text:
                try:
                    min_depth_value = float(im.text["iw3_min_depth_value"])
                    max_depth_value = float(im.text["iw3_min_depth_value"])
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
