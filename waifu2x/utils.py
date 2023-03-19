import os
from os import path
import torch
import torch.nn.functional as F
from nunif.transforms.tta import tta_merge, tta_split
from nunif.utils.render import tiled_render
from nunif.utils.alpha import AlphaBorderPadding
from nunif.models import load_model, get_model_config, data_parallel_model, call_model_method
from nunif.logger import logger


class Waifu2x():
    def __init__(self, model_dir, gpus):
        self.scale_model = None
        self.scale4x_model = None
        self.noise_models = [None] * 4
        self.noise_scale_models = [None] * 4
        self.noise_scale4x_models = [None] * 4
        if gpus[0] < 0:
            self.device = "cpu"
        else:
            if torch.cuda.is_available():
                self.device = f'cuda:{gpus[0]}'
            elif torch.backends.mps.is_available():
                device = f'mps:{device_ids[0]}'
            else:
                raise ValueError(f"No cuda/mps available. Use `--gpu -1` for CPU.")
        self.gpus = gpus
        self.model_dir = model_dir
        self.alpha_pad = AlphaBorderPadding()

    def _setup(self):
        if self.scale_model is not None:
            self.scale_model = self.scale_model.to(self.device)
            self.scale_model.eval()
        if self.scale4x_model is not None:
            self.scale4x_model = self.scale4x_model.to(self.device)
            self.scale4x_model.eval()

        for i in range(len(self.noise_models)):
            if self.noise_models[i] is not None:
                self.noise_models[i] = self.noise_models[i].to(self.device)
                self.noise_models[i].eval()

            if self.noise_scale_models[i] is not None:
                self.noise_scale_models[i] = self.noise_scale_models[i].to(self.device)
                self.noise_scale_models[i].eval()

            if self.noise_scale4x_models[i] is not None:
                self.noise_scale4x_models[i] = self.noise_scale4x_models[i].to(self.device)
                self.noise_scale4x_models[i].eval()

    def _load_model(self, method, noise_level):
        if method == "scale4x":
            if self.scale4x_model is not None:
                return
            if path.exists(path.join(self.model_dir, "scale4x.pth")):
                self.scale4x_model, _ = load_model(
                    path.join(self.model_dir, "scale4x.pth"),
                    map_location=self.device, device_ids=self.gpus)
            else:
                raise FileNotFoundError(f"scale4x.pth not found in {self.model_dir}")
        elif method == "scale":
            if self.scale_model is not None:
                return
            if path.exists(path.join(self.model_dir, "scale2x.pth")):
                self.scale_model, _ = load_model(
                    path.join(self.model_dir, "scale2x.pth"),
                    map_location=self.device, device_ids=self.gpus)
            else:
                if self.scale4x_model is None:
                    self._load_model("scale4x", noise_level)
                self.scale_model = data_parallel_model(call_model_method(self.scale4x_model, "to_2x"),
                                                       device_ids=self.gpus)
        elif method == "noise_scale4x":
            if self.noise_scale4x_models[noise_level] is not None:
                return
            if path.exists(path.join(self.model_dir, f"noise{noise_level}_scale4x.pth")):
                self.noise_scale4x_models[noise_level], _ = load_model(
                    path.join(self.model_dir, f"noise{noise_level}_scale4x.pth"),
                    map_location=self.device, device_ids=self.gpus)
            else:
                raise FileNotFoundError(f"scale4x.pth not found in {self.model_dir}")

        elif method == "noise_scale":
            if self.noise_scale_models[noise_level] is not None:
                return
            if path.exists(path.join(self.model_dir, f"noise{noise_level}_scale2x.pth")):
                self.noise_scale_models[noise_level], _ = load_model(
                    path.join(self.model_dir, f"noise{noise_level}_scale2x.pth"),
                    map_location=self.device, device_ids=self.gpus)
            else:
                if self.noise_scale4x_models[noise_level] is None:
                    self._load_model("noise_scale4x", noise_level)
                self.noise_scale_models[noise_level] = data_parallel_model(
                    call_model_method(self.noise_scale4x_models[noise_level], "to_2x"),
                    device_ids=self.gpus)
        elif method == "noise":
            if self.noise_models[noise_level] is not None:
                return
            if path.exists(path.join(self.model_dir, f"noise{noise_level}.pth")):
                self.noise_models[noise_level], _ = load_model(
                    path.join(self.model_dir, f"noise{noise_level}.pth"),
                    map_location=self.device, device_ids=self.gpus)
            else:
                if self.noise_scale4x_models[noise_level] is None:
                    self._load_model("noise_scale4x", noise_level)
                self.noise_models[noise_level] = data_parallel_model(
                    call_model_method(self.noise_scale4x_models[noise_level], "to_1x"),
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
                logger.warning(f"`scale4x_path used for alpha channel does not exist. "
                               "So use BILINEAR for upscaling alpha channel.")
        elif method == "noise_scale":
            self._load_model(method, noise_level)
            # for alpha channel
            try:
                self._load_model("scale", -1)
            except FileNotFoundError:
                logger.warning(f"`scale2x.pth` used for alpha channel does not exist. "
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
            return get_model_config(self.scale_model, "i2i_offset")
        elif method == "scale4x":
            return get_model_config(self.scale4x_model, "i2i_offset")
        elif method == "noise":
            return get_model_config(self.noise_models[noise_level], "i2i_offset")
        elif method == "noise_scale":
            return get_model_config(self.noise_scale_models[noise_level], "i2i_offset")
        elif method == "noise_scale4x":
            return get_model_config(self.noise_scale4x_models[noise_level], "i2i_offset")

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
            blank_alpha = torch.equal(alpha, torch.ones(alpha.shape, dtype=alpha.dtype))
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
