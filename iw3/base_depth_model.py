from abc import ABCMeta, abstractmethod
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.models.data_parallel import DeviceSwitchInference
import os
from os import path
import pickle
import torch
from nunif.device import create_device


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")


class BaseDepthModel(metaclass=ABCMeta):
    def __init__(self, model_type):
        self.device = None
        self.model = None
        self.model_type = model_type

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


def _test():
    pass


if __name__ == "__main__":
    _test()
