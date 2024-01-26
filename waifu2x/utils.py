from os import path
from packaging import version as packaging_version
import torch
import torch.nn.functional as F
from nunif.transforms.tta import tta_merge, tta_split
from nunif.utils.render import tiled_render
from nunif.utils.alpha import AlphaBorderPadding
from nunif.models import (
    load_model,
    data_parallel_model,
    compile_model, is_compiled_model,
)
from nunif.models.data_parallel import DataParallelInference
from nunif.device import create_device, autocast
from nunif.logger import logger
from nunif.utils.ui import HiddenPrints


# compling swin_unet model only works with torch >= 2.1.0
CAN_COMPILE_SWIN_UNET = packaging_version.parse(torch.__version__).release >= (2, 1, 0)
# torch 2.1.2 has a bug in F.scaled_dot_product_attention(mem efficient attention with mask)
CAN_COMPILE_WINC_UNET = False


def can_compile(model):
    if model is None:
        return False
    if isinstance(model, (torch.nn.DataParalle, DataParallelInference)):
        return False
    if not is_compiled_model(model):
        if model.name.startswith("waifu2x.swin_unet"):
            return CAN_COMPILE_SWIN_UNET
        elif model.name.startswith("waifu2x.winc_unet"):
            return CAN_COMPILE_WINC_UNET
        else:
            return True
    else:
        return False


class Waifu2x():
    def __init__(self, model_dir, gpus):
        self.scale_model = None
        self.scale4x_model = None
        self.noise_models = [None] * 4
        self.noise_scale_models = [None] * 4
        self.noise_scale4x_models = [None] * 4
        self.device = create_device(gpus)
        self.gpus = gpus
        self.model_dir = model_dir
        self.alpha_pad = AlphaBorderPadding()
        self.is_half = False

    def compile(self):
        # TODO: If dynamic tracing works well in the future,
        #       it is better to add `dynamic=True` for variable batch sizes.
        self._apply(lambda model: compile_model(model) if can_compile(model) else model)

    @torch.inference_mode()
    def warmup(self, tile_size, batch_size, enable_amp):
        models = [model for model in (self.scale_model, self.scale4x_model,
                                      *self.noise_models, *self.noise_scale_models,
                                      *self.noise_scale4x_models) if model is not None]
        for i, model in enumerate(models):
            for j, bs in enumerate(reversed(range(1, batch_size + 1))):
                x = torch.zeros((bs, 3, tile_size, tile_size),
                                device=self.device, dtype=torch.float16 if self.is_half else torch.float32)
                logger.debug(f"warmup {i * batch_size + j + 1}/{len(models) * batch_size}: {x.shape}")
                with autocast(device=self.device, enabled=enable_amp):
                    model(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

    def to(self, device):
        self.device = device
        self._setup()
        return self

    def half(self):
        self.is_half = True
        self._apply(lambda model: model.half())
        return self

    def float(self):
        self.is_half = False
        self._apply(lambda model: model.float())
        return self

    def _setup(self):
        self._apply(lambda model: model.to(self.device).eval())

    def _apply(self, func):
        if self.scale_model is not None:
            self.scale_model = func(self.scale_model)
        if self.scale4x_model is not None:
            self.scale4x_model = func(self.scale4x_model)

        for i in range(len(self.noise_models)):
            if self.noise_models[i] is not None:
                self.noise_models[i] = func(self.noise_models[i])

            if self.noise_scale_models[i] is not None:
                self.noise_scale_models[i] = func(self.noise_scale_models[i])

            if self.noise_scale4x_models[i] is not None:
                self.noise_scale4x_models[i] = func(self.noise_scale4x_models[i])

    def load_model_by_name(self, filename):
        with HiddenPrints():
            return load_model(path.join(self.model_dir, filename),
                              map_location=self.device, device_ids=self.gpus,
                              weights_only=True)[0]

    def has_model_file(self, filename):
        return path.exists(path.join(self.model_dir, filename))

    def _load_model(self, method, noise_level):
        if method == "scale4x":
            if self.scale4x_model is not None:
                return
            if self.has_model_file("scale4x.pth"):
                self.scale4x_model = self.load_model_by_name("scale4x.pth")
            else:
                raise FileNotFoundError(f"scale4x.pth not found in {self.model_dir}")
        elif method == "scale":
            if self.scale_model is not None:
                return
            if self.has_model_file("scale2x.pth"):
                self.scale_model = self.load_model_by_name("scale2x.pth")
            else:
                if self.scale4x_model is None:
                    self._load_model("scale4x", noise_level)
                self.scale_model = data_parallel_model(self.scale4x_model.to_2x(), device_ids=self.gpus)
        elif method == "noise_scale4x":
            if self.noise_scale4x_models[noise_level] is not None:
                return
            if self.has_model_file(f"noise{noise_level}_scale4x.pth"):
                self.noise_scale4x_models[noise_level] = self.load_model_by_name(f"noise{noise_level}_scale4x.pth")
            else:
                raise FileNotFoundError(f"noise{noise_level}_scale4x.pth not found in {self.model_dir}")

        elif method == "noise_scale":
            if self.noise_scale_models[noise_level] is not None:
                return
            if self.has_model_file(f"noise{noise_level}_scale2x.pth"):
                self.noise_scale_models[noise_level] = self.load_model_by_name(f"noise{noise_level}_scale2x.pth")
            else:
                if self.noise_scale4x_models[noise_level] is None:
                    self._load_model("noise_scale4x", noise_level)
                self.noise_scale_models[noise_level] = data_parallel_model(
                    self.noise_scale4x_models[noise_level].to_2x(),
                    device_ids=self.gpus)
        elif method == "noise":
            if self.noise_models[noise_level] is not None:
                return
            if self.has_model_file(f"noise{noise_level}.pth"):
                self.noise_models[noise_level] = self.load_model_by_name(f"noise{noise_level}.pth")
            else:
                if self.noise_scale4x_models[noise_level] is None:
                    self._load_model("noise_scale4x", noise_level)
                self.noise_models[noise_level] = data_parallel_model(
                    self.noise_scale4x_models[noise_level].to_1x(),
                    device_ids=self.gpus)
        else:
            raise ValueError(method)

    def load_model(self, method, noise_level):
        assert (method in ("scale", "noise_scale", "noise", "scale4x", "noise_scale4x"))
        assert (method in {"scale", "scale4x"} or 0 <= noise_level and noise_level < 4)

        if method in {"scale", "scale4x", "noise"}:
            self._load_model(method, noise_level)
        elif method == "noise_scale4x":
            self._load_model(method, noise_level)
            try:
                self._load_model("scale4x", -1)
            except FileNotFoundError:
                logger.warning("`scale4x_path used for alpha channel does not exist. "
                               "So use BILINEAR for upscaling alpha channel.")
        elif method == "noise_scale":
            self._load_model(method, noise_level)
            # for alpha channel
            try:
                self._load_model("scale", -1)
            except FileNotFoundError:
                logger.warning("`scale2x.pth` used for alpha channel does not exist. "
                               "So use BILINEAR for upscaling alpha channel.")
        self._setup()

    def load_model_all(self, load_4x=True):
        if load_4x:
            self._load_model("scale4x", -1)
            for noise_level in range(4):
                self._load_model("noise_scale4x", noise_level)

        self._load_model("scale", -1)
        for noise_level in range(4):
            self._load_model("noise_scale", noise_level)
            self._load_model("noise", noise_level)

        if not load_4x:
            # free 4x models
            self.scale4x_model = None
            self.noise_scale4x_models = [None] * 4
        self._setup()

    def render(self, x, method, noise_level, tile_size=256, batch_size=4, enable_amp=False):
        assert (method in ("scale", "noise_scale", "noise", "scale4x", "noise_scale4x"))
        assert (method in {"scale", "scale4x"} or 0 <= noise_level and noise_level < 4)
        if method == "scale":
            z = tiled_render(x, self.scale_model,
                             tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)
        elif method == "scale4x":
            z = tiled_render(x, self.scale4x_model,
                             tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)
        elif method == "noise":
            z = tiled_render(x, self.noise_models[noise_level],
                             tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)
        elif method == "noise_scale":
            z = tiled_render(x, self.noise_scale_models[noise_level],
                             tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)
        elif method == "noise_scale4x":
            z = tiled_render(x, self.noise_scale4x_models[noise_level],
                             tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)
        return z

    def _model_offset(self, method, noise_level):
        if method == "scale":
            return self.scale_model.i2i_offset
        elif method == "scale4x":
            return self.scale4x_model.i2i_offset
        elif method == "noise":
            return self.noise_models[noise_level].i2i_offset
        elif method == "noise_scale":
            return self.noise_scale_models[noise_level].i2i_offset
        elif method == "noise_scale4x":
            return self.noise_scale4x_models[noise_level].i2i_offset

    def convert(self, x, alpha, method, noise_level,
                tile_size=256, batch_size=4,
                tta=False, enable_amp=False):
        assert (not torch.is_grad_enabled())
        assert (x.shape[0] == 3)
        assert (alpha is None or alpha.shape[0] == 1 and alpha.shape[1:] == x.shape[1:])
        assert (method in ("scale", "scale4x", "noise_scale", "noise_scale4x", "noise"))
        assert (method in {"scale", "scale4x"} or 0 <= noise_level and noise_level < 4)

        if alpha is not None:
            # check all 1 alpha channel
            blank_alpha = torch.equal(alpha, torch.ones(alpha.shape, device=alpha.device, dtype=alpha.dtype))
        if alpha is not None and not blank_alpha:
            x = self.alpha_pad(x, alpha, self._model_offset(method, noise_level))
        if tta:
            rgb = tta_merge([
                self.render(xx, method, noise_level, tile_size, batch_size, enable_amp)
                for xx in tta_split(x)])
        else:
            rgb = self.render(x, method, noise_level, tile_size, batch_size, enable_amp)

        rgb = rgb.to("cpu")
        if alpha is not None and method in ("scale", "noise_scale", "scale4x", "noise_scale4x"):
            if not blank_alpha:
                model = self.scale4x_model if method in {"scale4x", "noise_scale4x"} else self.scale_model
                if model is not None:
                    alpha = alpha.expand(3, alpha.shape[1], alpha.shape[2])
                    alpha = tiled_render(alpha, model,
                                         tile_size=tile_size, batch_size=batch_size).mean(0, keepdim=True)
                else:
                    scale_factor = 4 if method in {"scale4x", "noise_scale4x"} else 2
                    alpha = F.interpolate(alpha.unsqueeze(0), scale_factor=scale_factor,
                                          mode="bilinear").squeeze(0)
            else:
                scale_factor = 4 if method in {"scale4x", "noise_scale4x"} else 2
                alpha = F.interpolate(alpha.unsqueeze(0), scale_factor=scale_factor, mode="nearest").squeeze(0)
            alpha = alpha.to("cpu")

        return rgb, alpha
