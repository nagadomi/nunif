import contextlib
import os
import pickle
import threading
from abc import ABCMeta, abstractmethod
from typing import Any, overload

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import functional as TF

from nunif.device import create_device
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.models.utils import compile_function, compile_model
from nunif.utils.downloader import CheckpointDownloader
from nunif.utils.ui import HiddenPrints, TorchHubDir

from .depth_scaler import EMAMinMaxScaler
from .hub_dir import HUB_MODEL_DIR


class _ForceBatchSizeContext:
    def __init__(self, base_model: "BaseDepthModel", batch_size: int | None):
        if batch_size is not None:
            assert batch_size > 0
        self.base_model = base_model
        self.batch_size = batch_size
        self.prev_batch_size: int | None = None

    def __enter__(self):
        self.prev_batch_size = self.base_model._force_batch_size
        self.base_model._force_batch_size = self.batch_size
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model._force_batch_size = self.prev_batch_size
        return False


class _CompileContext:
    def __init__(self, base_model: "BaseDepthModel", force_batch_size: int | None = None):
        self.base_model = base_model
        self.force_batch_size_ctx = (
            _ForceBatchSizeContext(base_model, force_batch_size)
            if force_batch_size is not None and not isinstance(base_model.model, DeviceSwitchInference)
            else None
        )

    def __enter__(self):
        self.base_model.compile()
        if self.force_batch_size_ctx is not None:
            self.force_batch_size_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.force_batch_size_ctx is not None:
            self.force_batch_size_ctx.__exit__(exc_type, exc_val, exc_tb)
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
    def update_hub(cls):
        for subcls in cls.get_registered_classes():
            subcls.force_update()

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
    def is_matting(cls, model_type: str | None = None) -> bool:
        if cls is BaseDepthModel:
            if model_type is None:
                return False
            subcls = cls.get_depth_model_class(model_type)
            return subcls.is_matting(model_type) if subcls is not None else False
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
        for subcls in cls.get_registered_classes():
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
        self.depth_aa = None
        self._depth_aa_infer_backup = None
        self.model_type = model_type
        self.scaler = self.create_depth_scaler()
        self.limit_resolution = False
        self._force_batch_size: int | None = None

    def create_depth_scaler(self):
        # This can be overridden
        return EMAMinMaxScaler(decay=0, buffer_size=1)

    def force_batch_size(self, batch_size: int | None) -> _ForceBatchSizeContext:
        return _ForceBatchSizeContext(self, batch_size)

    def compile_context(self, enabled: bool = True, force_batch_size: int | None = None):
        if enabled:
            return _CompileContext(self, force_batch_size=force_batch_size)
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

    @classmethod
    def get_model_url(cls, model_type: str) -> str | None:
        return None

    @classmethod
    def download(
        cls,
        model_type: str,
        stop_event: threading.Event | None = None,
        tqdm_fn: Any = None,
    ) -> bool:
        if cls.has_checkpoint_file(model_type):
            return True
        url = cls.get_model_url(model_type)
        if url is None:
            return True
        dst_path = cls.get_model_path(model_type)
        downloader = CheckpointDownloader(url=url, dst_path=dst_path, name=f"Downloading {model_type}")
        return downloader.run(stop_event=stop_event, tqdm_fn=tqdm_fn)

    @abstractmethod
    def is_metric(self):
        pass

    def is_image_supported(self):
        return True

    def is_video_supported(self):
        return True

    def is_streaming(self):
        return False

    def is_buffered(self) -> bool:
        return self.get_ema_buffer_size() > 1

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

    def load(
        self,
        gpu=0,
        resolution=None,
        limit_resolution=False,
        stop_event: threading.Event | None = None,
        tqdm_fn: Any = None,
        **kwargs,
    ):
        if not self.download(self.model_type, stop_event=stop_event, tqdm_fn=tqdm_fn):
            return self

        self.device = create_device(gpu)
        self.limit_resolution = limit_resolution

        with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
            try:
                self.model = self.load_model(self.model_type, resolution=resolution, device=self.device, **kwargs)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    do_handle = "PytorchStreamReader" in repr(e)
                elif isinstance(e, pickle.PickleError) or e.__class__.__name__ == "SafetensorError":
                    do_handle = True
                else:
                    do_handle = False
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

    def compile_depth_aa(self) -> None:
        if self._depth_aa_infer_backup is None and self.depth_aa is not None:
            self._depth_aa_infer_backup = self.depth_aa.infer
            self.depth_aa.infer = compile_function(self.depth_aa.infer, device=self.device)

    def clear_compiled_depth_aa(self) -> None:
        if self._depth_aa_infer_backup is not None:
            if self.depth_aa is not None:
                self.depth_aa.infer = self._depth_aa_infer_backup
            self._depth_aa_infer_backup = None

    def compile(self):
        if self.model_backup is None and not isinstance(self.model, DeviceSwitchInference):
            self.model_backup = self.model
            self.model = compile_model(self.model)
        self.compile_depth_aa()

    def clear_compiled_model(self):
        if self.model_backup is not None:
            self.model = self.model_backup
            self.model_backup = None
        self.clear_compiled_depth_aa()

    @abstractmethod
    def infer(
        self,
        x: torch.Tensor | Image.Image,
        tta: bool = False,
        low_vram: bool = False,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        alpha: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        pass

    @overload
    def _prepare_infer_input(
        self,
        x: torch.Tensor | Image.Image,
        alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool, int]: ...

    @overload
    def _prepare_infer_input(
        self,
        x: torch.Tensor | Image.Image,
        alpha: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None], bool, int]: ...

    def _prepare_infer_input(
        self,
        x: torch.Tensor | Image.Image,
        alpha: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor | None] | None, bool, int]:
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        assert torch.is_tensor(x)

        is_batch = x.ndim == 4
        if x.ndim == 3:
            x = x.unsqueeze(0)
        assert x.ndim == 4

        orig_batch_size = x.shape[0]
        assert orig_batch_size > 0
        target_batch_size = self._force_batch_size
        if target_batch_size is not None and not self.is_streaming():
            assert target_batch_size > 0
            assert orig_batch_size <= target_batch_size
            if orig_batch_size < target_batch_size:
                pad_n = target_batch_size - orig_batch_size
                # Preserve x.is_inference() to avoid DispatchKeySet recompilation in torch.compile
                with torch.inference_mode(x.is_inference()), torch.no_grad():
                    x = torch.cat([x, x[-1:].expand(pad_n, *x.shape[1:])], dim=0)
                if alpha is not None:
                    if isinstance(alpha, list):
                        assert len(alpha) == orig_batch_size
                        alpha = alpha + [alpha[-1]] * pad_n
                    elif torch.is_tensor(alpha):
                        if alpha.ndim == 3:
                            alpha = alpha.unsqueeze(0)
                        assert alpha.shape[0] == orig_batch_size
                        with torch.inference_mode(alpha.is_inference()), torch.no_grad():
                            alpha = torch.cat([alpha, alpha[-1:].expand(pad_n, *alpha.shape[1:])], dim=0)

        return x, alpha, is_batch, orig_batch_size

    @overload
    @staticmethod
    def _restore_infer_output(out: torch.Tensor, is_batch: bool, orig_batch_size: int) -> torch.Tensor: ...

    @overload
    @staticmethod
    def _restore_infer_output(
        out: tuple[torch.Tensor, torch.Tensor], is_batch: bool, orig_batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def _restore_infer_output(
        out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        is_batch: bool,
        orig_batch_size: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def _restore_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] > orig_batch_size:
                t = t[:orig_batch_size]
            if not is_batch:
                assert t.shape[0] == 1
                t = t.squeeze(0)
            return t

        if isinstance(out, tuple):
            return tuple(_restore_tensor(t) for t in out)  # type: ignore[return-value]
        return _restore_tensor(out)

    @staticmethod
    def apply_alpha_mask(
        depth: torch.Tensor,
        alpha: torch.Tensor | list[torch.Tensor | None] | None,
        flip_aug: bool = False,
        is_metric: bool = False,
    ) -> torch.Tensor:
        if alpha is None:
            return depth

        if isinstance(alpha, list):
            if all(a is None for a in alpha):
                return depth
            squeeze = False
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
                squeeze = True
            for i, a in enumerate(alpha):
                if a is not None and i < depth.shape[0]:
                    depth[i : i + 1] = BaseDepthModel.apply_alpha_mask(
                        depth[i : i + 1], a, flip_aug=False, is_metric=is_metric
                    )
            if squeeze:
                depth = depth.squeeze(0)
            return depth

        squeeze = False
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
            squeeze = True

        if alpha.device != depth.device:
            alpha = alpha.to(depth.device)
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(0)

        if alpha.shape[-2:] != depth.shape[-2:]:
            alpha = F.interpolate(
                alpha,
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        if flip_aug and depth.shape[0] == alpha.shape[0] * 2:
            alpha = torch.cat([alpha, torch.flip(alpha, dims=[3])], dim=0)

        mask = alpha > 0.0
        num_pixels = mask.shape[-2] * mask.shape[-1]
        fg_count = mask.sum(dim=(-2, -1), keepdim=True)
        has_partial_alpha = (fg_count > 0) & (fg_count < num_pixels)
        if is_metric:
            fg_depth = torch.where(mask, depth, torch.tensor(-float("inf"), device=depth.device, dtype=depth.dtype))
            fill_val = (fg_depth.amax(dim=(-2, -1), keepdim=True) + depth.amax(dim=(-2, -1), keepdim=True)) * 0.5
        else:
            fg_depth = torch.where(mask, depth, torch.tensor(float("inf"), device=depth.device, dtype=depth.dtype))
            fill_val = (fg_depth.amin(dim=(-2, -1), keepdim=True) + depth.amin(dim=(-2, -1), keepdim=True)) * 0.5
        bg_val = torch.where(has_partial_alpha, fill_val, depth)
        depth = torch.where(mask, depth, bg_val)

        if squeeze:
            depth = depth.squeeze(0)
        return depth

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
    def to_uint16_numpy(depth: torch.Tensor) -> np.ndarray:
        depth = torch.clamp(depth, 0, 1)
        return (0xFFFF * depth).round().to(torch.uint16).squeeze(0).cpu().numpy()

    @staticmethod
    def normalized_depth_to_uint16_image(depth: torch.Tensor | np.ndarray) -> Image.Image:
        if isinstance(depth, np.ndarray):
            return Image.fromarray(depth)
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

        im.save(file_path, pnginfo=metadata, compress_level=1)

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
