from os import path
import torch
import torchvision.transforms.functional as TF
from ... transforms import functional as NF
from ... utils import tiled_render, make_alpha_border
from ... models import load_model


class Waifu2x():
    def __init__(self, model_dir, gpus):
        self.scale_model = None
        self.noise_models = [None] * 4
        self.noise_scale_models = [None] * 4
        if gpus[0] < 0:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{gpus[0]}'
        self.gpus = gpus
        self.model_dir = model_dir

    def _setup(self):
        if self.scale_model is not None:
            self.scale_model = self.scale_model.to(self.device)
            self.scale_model.eval()
            if len(self.gpus) > 1:
                self.scale_model = torch.nn.DataParallel(self.scale_model, device_ids=self.gpus)
        for i in range(len(self.noise_models)):
            if self.noise_models[i] is not None:
                self.noise_models[i] = self.noise_models[i].to(self.device)
                self.noise_models[i].eval()
                if len(self.gpus) > 1:
                    self.noise_models[i] = torch.nn.DataParallel(self.noise_models[i], device_ids=self.gpus)

            if self.noise_scale_models[i] is not None:
                self.noise_scale_models[i] = self.noise_scale_models[i].to(self.device)
                self.noise_scale_models[i].eval()
                if len(self.gpus) > 1:
                    self.noise_scale_models[i] = torch.nn.DataParallel(self.noise_scale_models[i], device_ids=self.gpus)

    def load_model(self, method, noise_level):
        assert(method in ("scale", "noise_scale", "noise"))
        assert(0 <= noise_level and noise_level < 4)
        if method == "scale":
            self.scale_model = load_model(path.join(self.model_dir, "scale2x.pth"))
        elif method == "noise":
            self.noise_models[noise_level] = load_model(path.join(self.model_dir, f"noise{noise_level}.pth"))
        elif method == "noise_scale":
            self.scale_model = load_model(path.join(self.model_dir, "scale2x.pth"))  # for alpha channel
            self.noise_scale_models[noise_level] = load_model(path.join(self.model_dir, f"noise{noise_level}_scale2x.pth"))
        self._setup()

    def load_model_all(self):
        self.scale_model = load_model(path.join(self.model_dir, "scale2x.pth"))
        self.noise_models = [load_model(path.join(self.model_dir, f"noise{noise_level}.pth"))
                             for noise_level in range(4)]
        self.noise_scale_models = [load_model(path.join(self.model_dir, f"noise{noise_level}_scale2x.pth"))
                                   for noise_level in range(4)]
        self._setup()

    def convert_(self, x, method, noise_level, tile_size=256, batch_size=4):
        assert(method in ("scale", "noise_scale", "noise"))
        assert(0 <= noise_level and noise_level < 4)
        if method == "scale":
            z = tiled_render(x, self.scale_model, self.device,
                             tile_size=tile_size, batch_size=batch_size)
        elif method == "noise":
            z = tiled_render(x, self.noise_models[noise_level], self.device,
                             tile_size=tile_size, batch_size=batch_size)
        elif method == "noise_scale":
            z = tiled_render(x, self.noise_scale_models[noise_level],
                             self.device, tile_size=tile_size, batch_size=batch_size)
        return z

    def _model_offset(self, method, noise_level):
        if method == "scale":
            return self.scale_model.offset
        elif method == "noise":
            return self.noise_models[noise_level].offset
        elif method == "noise_scale":
            return self.noise_scale_models[noise_level].offset

    def convert(self, im, meta, method, noise_level, tile_size=256, batch_size=4, tta=False):
        assert(method in ("scale", "noise_scale", "noise"))
        assert(0 <= noise_level and noise_level < 4)

        x = TF.to_tensor(im)
        alpha = None
        if "alpha" in meta:
            alpha = TF.to_tensor(meta["alpha"])
            x = make_alpha_border(x, alpha, self._model_offset(method, noise_level))

        if tta:
            rgb = NF.tta_merge([self.convert_(x_, method, noise_level, tile_size, batch_size) for x_ in NF.tta_split(x)])
        else:
            rgb = self.convert_(x, method, noise_level, tile_size, batch_size)
        rgb = TF.to_pil_image(NF.quantize256(rgb).to("cpu"))

        if alpha is not None and method in ("scale", "noise_scale"):
            alpha = alpha.expand(3, alpha.shape[1], alpha.shape[2])
            alpha = tiled_render(alpha, self.scale_model, self.device,
                                 tile_size=tile_size, batch_size=batch_size).mean(0)
        if alpha is not None:
            alpha = TF.to_pil_image(NF.quantize256(alpha).to("cpu"))
            rgb.putalpha(alpha)

        return rgb
