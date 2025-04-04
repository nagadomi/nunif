from os import path
import torch
from .utils import Waifu2x
from .download_models import main as download_main
from nunif.utils import pil_io
import PIL


MODEL_DIR = path.join(path.dirname(path.abspath(__file__)), "pretrained_models")
MODEL_TYPES = {
    # default models.
    "art": path.join(MODEL_DIR, "swin_unet", "art"),
    "art_scan": path.join(MODEL_DIR, "swin_unet", "art_scan"),
    "photo": path.join(MODEL_DIR, "swin_unet", "photo"),
    # arch
    "swin_unet/art": path.join(MODEL_DIR, "swin_unet", "art"),
    "swin_unet/art_scan": path.join(MODEL_DIR, "swin_unet", "art_scan"),
    "swin_unet/photo": path.join(MODEL_DIR, "swin_unet", "photo"),
    "cunet/art": path.join(MODEL_DIR, "cunet", "art"),
    "upconv_7/art": path.join(MODEL_DIR, "upconv_7", "art"),
    "upconv_7/photo": path.join(MODEL_DIR, "upconv_7", "photo"),
}
NO_4X_MODELS = {"cunet/art", "upconv_7/art", "upconv_7/photo"}
METHODS = [
    "noise", "scale", "noise_scale",
    "scale2x", "noise_scale2x",
    "scale4x", "noise_scale4x"
]


class Waifu2xImageModel():
    def __init__(self, model_type, method=None, noise_level=-1,
                 device_ids=[-1], tile_size=None, batch_size=None, keep_alpha=True, amp=True):
        self.model_type = model_type
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.keep_alpha = keep_alpha
        self.amp = amp
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type: choose from {list(MODEL_TYPES.keys())}")
        if method is not None and method not in METHODS:
            raise ValueError(f"method: choose from {METHODS}")
        if method is not None and method.startswith("noise") and noise_level not in {0, 1, 2, 3}:
            raise ValueError("noise_level: choose from [0, 1, 2, 3]")

        self.ctx = Waifu2x(MODEL_TYPES[model_type], device_ids)
        if method is not None:
            method = self.normalize_method(method, noise_level)
            self.ctx.load_model(method, noise_level)
            self.set_mode(method, noise_level)
        else:
            self.method = None
            self.noise_level = None
            self.ctx.load_model_all(load_4x=(model_type not in NO_4X_MODELS))

    def set_mode(self, method, noise_level=-1):
        method = self.normalize_method(method, noise_level)
        if self.model_type in NO_4X_MODELS and method in {"scale4x", "noise_scale4x"}:
            raise ValueError(f"method: {self.model_type} does not support {method}")
        if (method in {"noise", "noise_scale4x", "noise_scale", "noise_scale2x"} and
                noise_level not in {0, 1, 2, 3}):
            raise ValueError("noise_level: choose from (0, 1, 2, 3)")
        self.method = method
        self.noise_level = noise_level

    def compile(self):
        self.ctx.compile()
        return self

    def to(self, device):
        self.ctx = self.ctx.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def half(self):
        self.ctx.half()
        return self

    def float(self):
        self.ctx.float()
        return self

    @property
    def is_half(self):
        return self.ctx.is_half

    @property
    def device(self):
        return self.ctx.device

    def convert(self, input_filepath, output_filepath, tta=False, format="png", **kwargs):
        im, meta = pil_io.load_image(input_filepath, keep_alpha=self.keep_alpha)
        new_im = self.infer_pil(im, tta=tta, **kwargs)
        pil_io.save_image(new_im, output_filepath, meta=meta, format=format)

    def infer_file(self, filepath, tta=False, output_type="pil", **kwargs):
        im, meta = pil_io.load_image(filepath, keep_alpha=self.keep_alpha)
        return self.infer_pil(im, tta=tta, output_type=output_type, **kwargs)

    def infer_pil(self, pil_image, tta=False, output_type="pil", **kwargs):
        if self.keep_alpha:
            rgb, alpha = pil_io.to_tensor(pil_image, return_alpha=self.keep_alpha)
        else:
            rgb = pil_io.to_tensor(pil_image, return_alpha=self.keep_alpha)
            alpha = None

        rgb = rgb.to(self.device)
        if self.is_half:
            rgb = rgb.half()
        if alpha is not None:
            alpha = alpha.to(self.device)
            if self.is_half:
                alpha = alpha.half()

        return self.infer_tensor(rgb, alpha, tta=tta, output_type=output_type, **kwargs)

    def infer_tensor(self, rgb, alpha=None, tta=False, output_type="pil", **kwargs):
        method = kwargs.get("method", self.method)
        noise_level = kwargs.get("noise_level", self.noise_level)
        if method is None:
            raise ValueError(("method is None. Call `model.set_mode(method, noise_level)`"
                              " or use method and noise_level kwargs"))
        with torch.inference_mode():
            rgb, alpha = self.ctx.convert(
                rgb, alpha, method, noise_level,
                tile_size=self.tile_size, batch_size=self.batch_size,
                tta=tta, enable_amp=self.amp)
        if output_type == "tensor":
            return (rgb, alpha)
        else:
            return pil_io.to_image(rgb, alpha)

    def infer(self, x, tta=False, output_type="pil", **kwargs):
        if isinstance(x, str):
            return self.infer_file(x, tta=tta, output_type=output_type, **kwargs)
        if isinstance(x, PIL.Image.Image):
            return self.infer_pil(x, tta=tta, output_type=output_type, **kwargs)
        elif torch.is_tensor(x):
            return self.infer_tensor(x, tta=tta, output_type=output_type, **kwargs)
        else:
            raise ValueError("Unsupported input format")

    def __call__(self, x, tta=False, output_type="pil", **kwargs):
        return self.infer(x, tta=tta, output_type=output_type, **kwargs)

    @staticmethod
    def normalize_method(method, noise_level):
        if method is None:
            return None
        if method == "scale2x":
            method = "scale"
        if method == "noise_scale2x":
            method = "noise_scale"
        if method == "scale" and noise_level >= 0:
            method = "noise_scale"
        if method == "scale4x" and noise_level >= 0:
            method = "noise_scale4x"
        return method


def waifu2x(model_type="art",
            method=None, noise_level=-1,
            device_ids=[-1], tile_size=None, batch_size=None, keep_alpha=True, amp=True,
            **kwargs):
    download_main()
    return Waifu2xImageModel(
        model_type=model_type,
        method=method, noise_level=noise_level,
        device_ids=device_ids, tile_size=tile_size, batch_size=batch_size,
        keep_alpha=keep_alpha, amp=amp)


def _test():
    import os
    import argparse
    import threading

    ROOT_DIR = path.join(path.dirname(__file__), "..")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    im = PIL.Image.open(args.input)
    # model_type
    for model_type in MODEL_TYPES.keys():
        for method in ("scale2x", "scale4x"):
            if method == "scale4x" and model_type in NO_4X_MODELS:
                continue
            # Load a model with fixed method and noise_level
            model = torch.hub.load(
                ROOT_DIR, "waifu2x", keep_alpha=True,
                model_type=model_type, method=method, noise_level=3,
                source="local", trust_repo=True)
            model = model.to("cuda")
            out = model(im)
            out.save(path.join(args.output, f"{model_type.replace('/', '-')}_{method}.png"))

    # Load all method and noise_level models
    lock = threading.RLock()
    model = torch.hub.load(
        ROOT_DIR, "waifu2x", keep_alpha=True,
        model_type="art_scan",
        source="local", trust_repo=True).to("cuda")
    for noise_level in (0, 1, 2, 3):
        with lock:  # model.set_mode -> model.infer block is not thread-safe, so lock
            # Select method and noise_level
            model.set_mode("scale", noise_level)
            out = model(im)
        out.save(path.join(args.output, f"noise_scale_{noise_level}.png"))
    for noise_level in (0, 1, 2, 3):
        out = model(im, method="noise", noise_level=noise_level)
        out.save(path.join(args.output, f"noise_{noise_level}.png"))


def _test_device_memory():
    from time import time

    ROOT_DIR = path.join(path.dirname(__file__), "..")
    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art",
        source="local", trust_repo=True)

    # load model to gpu memory
    t = time()
    model = model.cuda()
    print("* load", round(time() - t, 4))
    print(torch.cuda.memory_summary())

    # release gpu memory
    t = time()
    model = model.cpu()
    torch.cuda.empty_cache()
    print("* release", round(time() - t, 4))
    print(torch.cuda.memory_summary())

    # reload model to gpu memory
    t = time()
    model = model.cuda()
    print("* reload", round(time() - t, 4))
    print(torch.cuda.memory_summary())


def _test_tensor_input():
    import argparse
    import torchvision.transforms.functional as TF

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file")
    args = parser.parse_args()

    im = PIL.Image.open(args.input)
    x = TF.to_tensor(im)

    ROOT_DIR = path.join(path.dirname(__file__), "..")
    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3, amp=True,
        source="local", trust_repo=True)

    model = model.cuda()
    gt = model.infer(x.cuda())
    for device in ("cpu", "cuda"):
        model = model.to(device)
        for tensor in (x.cuda(), x.cpu(), x.half().cuda(), x.half().cpu()):
            ret = model.infer(tensor.float().to(model.device))
            mean_diff = (TF.to_tensor(gt) - TF.to_tensor(ret)).abs().mean()
            print(model.device, tensor.dtype, tensor.device, mean_diff)

    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3, amp=False,
        source="local", trust_repo=True)
    model = model.cuda().half()
    for tensor in (x.cuda(), x.cuda().half()):
        ret = model.infer(tensor.to(model.device).half())
        mean_diff = (TF.to_tensor(gt) - TF.to_tensor(ret)).abs().mean()
        print(model.device, tensor.dtype, tensor.device, mean_diff)


def _test_amp():
    import argparse
    from time import time
    import torchvision.transforms.functional as TF

    ROOT_DIR = path.join(path.dirname(__file__), "..")
    LOOP_N = 10

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file")
    args = parser.parse_args()
    im = PIL.Image.open(args.input)

    # AMP=True
    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3,
        source="local", trust_repo=True).cuda()
    t = time()
    for i in range(LOOP_N):
        amp_t_im = model(im)
    print("amp=True", round((time() - t) / LOOP_N, 4), "sec / image")

    # AMP=False
    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3, amp=False,
        source="local", trust_repo=True).cuda()
    t = time()
    for i in range(LOOP_N):
        amp_f_im = model(im)
    print("amp=False", round((time() - t) / LOOP_N, 4), "sec / image")

    # AMP=False, manual setting float16
    # NOTE: `torch.set_default_dtype(torch.float16)` does not work when `torch.hub.load`
    if False:
        torch.set_default_dtype(torch.float16)
    model = model.half()
    t = time()
    for i in range(LOOP_N):
        amp_f_half_im = model(im)
    print("amp=False half()", round((time() - t) / LOOP_N, 4), "sec / image")

    def image_diff(a, b):
        return (TF.to_tensor(a) - TF.to_tensor(b)).abs().mean()

    print("image_diff(amp=True, amp=False)", image_diff(amp_t_im, amp_f_im))
    print("image_diff(amp=True, amp=False half())", image_diff(amp_t_im, amp_f_half_im))


def _test_half():
    import argparse
    import torchvision.transforms.functional as TF

    ROOT_DIR = path.join(path.dirname(__file__), "..")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file")
    args = parser.parse_args()
    im = PIL.Image.open(args.input)

    model = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3, amp=False,
        source="local", trust_repo=True).cuda()
    ret = model(im)

    model = model.half()
    ret_half = model(im)
    print((TF.to_tensor(ret) - TF.to_tensor(ret_half)).abs().mean())


def _test_alias():
    import argparse
    import torchvision.transforms.functional as TF

    ROOT_DIR = path.join(path.dirname(__file__), "..")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input file")
    args = parser.parse_args()
    im = PIL.Image.open(args.input)

    model1 = torch.hub.load(
        ROOT_DIR, "waifu2x", model_type="art", method="scale", noise_level=3, amp=False,
        source="local", trust_repo=True).cuda()
    ret1 = model1(im)
    model2 = torch.hub.load(
        ROOT_DIR, "superresolution", model_type="art", method="scale", noise_level=3, amp=False,
        source="local", trust_repo=True).cuda()
    ret2 = model2(im)
    print((TF.to_tensor(ret1) - TF.to_tensor(ret2)).abs().sum())


if __name__ == "__main__":
    _test()
    # _test_device_memory()
    # _test_tensor_input()
    # _test_amp()
    # _test_half()
    # _test_alias()
