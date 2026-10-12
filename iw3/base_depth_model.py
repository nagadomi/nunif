import contextlib
import os
import pickle
from abc import ABCMeta, abstractmethod

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import functional as TF

from nunif.device import create_device
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.models.utils import compile_model
from nunif.utils.ui import HiddenPrints, TorchHubDir

from .depth_scaler import EMAMinMaxScaler
from .hub_dir import HUB_MODEL_DIR


class _CompileContext:
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False


class BaseDepthModel(metaclass=ABCMeta):
    _subclasses: list[type["BaseDepthModel"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls not in BaseDepthModel._subclasses:
            BaseDepthModel._subclasses.append(cls)

    order: int = 100

    @classmethod
    def get_registered_classes(cls) -> list[type["BaseDepthModel"]]:
        from . import depth_models  # noqa: F401

        return sorted(cls._subclasses, key=lambda c: getattr(c, "order", 100))

    @classmethod
    def get_all_model_types(cls) -> list[str]:
        model_types: list[str] = []
        for subcls in cls.get_registered_classes():
            for name in subcls.get_model_types():
                if name not in model_types:
                    model_types.append(name)
        return model_types

    @classmethod
    def has_auto_download(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.has_auto_download(model_type) if subcls is not None else False
        return False

    @classmethod
    def is_small(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_small(model_type) if subcls is not None else False
        return False

    @classmethod
    def is_desktop_supported(cls) -> bool:
        return True

    @classmethod
    def is_model_available(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_model_available(model_type) if subcls is not None else False
        return cls.has_auto_download(model_type) or cls.has_checkpoint_file(model_type)

    @classmethod
    def get_available_models(cls, desktop: bool = False, small_only: bool = False) -> list[str]:
        models: list[str] = []
        for subcls in cls.get_registered_classes():
            if desktop and not subcls.is_desktop_supported():
                continue
            for m in subcls.get_model_types():
                if m == "NULL":
                    continue
                if small_only and not subcls.is_small(m):
                    continue
                if subcls.is_model_available(m):
                    models.append(m)
        return models

    @classmethod
    def get_depth_model_class(cls, model_type: str) -> type["BaseDepthModel"] | None:
        for subcls in cls._subclasses:
            if subcls.supported(model_type):
                return subcls
        return None

    @classmethod
    def has_model_checkpoint_file(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.has_checkpoint_file(model_type) if subcls is not None else False
        return cls.has_checkpoint_file(model_type)

    @classmethod
    def is_depth_aa_supported(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_depth_aa_supported(model_type) if subcls is not None else False
        return False

    @classmethod
    def is_resolution_change_supported(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_resolution_change_supported(model_type) if subcls is not None else True
        return True

    @classmethod
    def is_limit_resolution_supported(cls, model_type: str) -> bool:
        if cls is BaseDepthModel:
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_limit_resolution_supported(model_type) if subcls is not None else True
        return True

    def __init__(self, model_type):
        self.device = None
        self.model = None
        self.model_backup = None  # for compile
        self.model_type = model_type
        self.scaler = self.create_depth_scaler()
        self.limit_resolution = False

    def create_depth_scaler(self):
        # This can be overridden
        return EMAMinMaxScaler(decay=0, buffer_size=1)

    def compile_context(self, enabled=True):
        if enabled:
            return _CompileContext(self)
        else:
            return contextlib.nullcontext()

    @classmethod
    @abstractmethod
    def get_name(cls):
        pass

    def loaded(self):
        return self.model is not None

    @classmethod
    @abstractmethod
    def supported(cls, model_type):
        pass

    @classmethod
    @abstractmethod
    def get_model_types(cls) -> list[str]:
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

    def is_streaming(self):
        return False

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

    def load(self, gpu=0, resolution=None, limit_resolution=False, **kwargs):
        self.device = create_device(gpu)
        self.limit_resolution = limit_resolution

        with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
            try:
                self.model = self.load_model(self.model_type, resolution=resolution, device=self.device, **kwargs)
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

        if isinstance(gpu, (list, tuple)) and len(gpu) > 1:
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
    def infer(
        self,
        x: torch.Tensor | Image.Image,
        tta: bool = False,
        low_vram: bool = False,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        pass

    def flush(
        self,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        return None

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
        if depth is None:
            return []
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
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
    def normalized_depth_to_uint16_image(depth: torch.Tensor) -> Image.Image:
        depth = torch.clamp(depth, 0, 1)
        depth_int = (0xFFFF * depth).round().to(torch.uint16).squeeze(0).cpu().numpy()
        return Image.fromarray(depth_int)

    @staticmethod
    def save_normalized_depth(depth, file_path, png_info={}, min_depth_value=None, max_depth_value=None):
        if min_depth_value is not None:
            png_info.update(iw3_min_depth_value=float(min_depth_value))
        if max_depth_value is not None:
            png_info.update(iw3_max_depth_value=float(max_depth_value))

        im = BaseDepthModel.normalized_depth_to_uint16_image(depth)
        metadata = PngInfo()
        for k, v in png_info.items():
            metadata.add_text(k, str(v))

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
                depth = torch.clamp(depth.to(torch.float32) / 0xFFFF, 0, 1)
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
