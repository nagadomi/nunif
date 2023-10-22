import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.device import create_device, autocast
from nunif.models.data_parallel import DataParallelInference


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")


def load_model(model_type="ZoeD_N", gpu=0, height=None):
    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        if not os.getenv("IW3_DEBUG"):
            model = torch.hub.load("nagadomi/ZoeDepth_iw3:main", model_type, config_mode="infer",
                                   pretrained=True, verbose=False, trust_repo=True)
        else:
            assert path.exists("../ZoeDepth_iw3/hubconf.py")
            model = torch.hub.load("../ZoeDepth_iw3", model_type, source="local", config_mode="infer",
                                   pretrained=True, verbose=False, trust_repo=True)
    device = create_device(gpu)
    model = model.to(device).eval()
    if isinstance(gpu, (list, tuple)) and len(gpu) > 1:
        model = DataParallelInference(model, device_ids=gpu)
    if height is not None:
        model.core.prep.resizer = HeightResizer(height, height)
    else:
        model.core.prep.resizer = HeightResizer()

    return model


ZOED_MODEL_FILES = {
    "ZoeD_N": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_N.pt"),
    "ZoeD_K": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_K.pt"),
    "ZoeD_NK": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_NK.pt"),
}


def has_model(model_type="ZoeD_N"):
    assert model_type in ZOED_MODEL_FILES
    return path.exists(ZOED_MODEL_FILES[model_type])


def force_update_midas():
    with TorchHubDir(HUB_MODEL_DIR):
        torch.hub.help("nagadomi/MiDaS_iw3:master", "DPT_BEiT_L_384", force_reload=True, trust_repo=True)


def force_update_zoedepth():
    with TorchHubDir(HUB_MODEL_DIR):
        torch.hub.help("nagadomi/ZoeDepth_iw3:main", "ZoeD_N", force_reload=True, trust_repo=True)


def _forward(model, x, enable_amp):
    with autocast(device=x.device, enabled=enable_amp):
        out = model(x)['metric_depth']
    return out


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, int16=True, enable_amp=False,
                output_device="cpu", device=None):
    # _patch_resize_debug(model)
    device = device if device is not None else model.device
    batch = False
    if torch.is_tensor(im):
        assert im.ndim == 3 or im.ndim == 4
        if im.ndim == 3:
            im = im.unsqueeze(0)
        else:
            batch = True
        x = im.to(device)
    else:
        # PIL
        x = TF.to_tensor(im).unsqueeze(0).to(device)

    def get_pad(x):
        pad_base_h = int((x.shape[2] * 0.5) ** 0.5 * 3)
        pad_base_w = int((x.shape[3] * 0.5) ** 0.5 * 3)
        if x.shape[2] > x.shape[3]:
            diff = (x.shape[2] + pad_base_h * 2 - x.shape[3] + pad_base_w * 2)
            pad_w1 = diff // 2
            pad_w2 = diff - pad_w1
            pad_w1 += pad_base_w
            pad_w2 += pad_base_w
            pad_h1 = pad_h2 = pad_base_h
        else:
            pad_w1 = pad_w2 = pad_base_w
            pad_h1 = pad_h2 = pad_base_h
        return (min(pad_w1, x.shape[3] - 1), min(pad_w2, x.shape[3] - 1),
                min(pad_h1, x.shape[2] - 1), min(pad_h2, x.shape[2] - 1))

    if not low_vram:
        if flip_aug:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        pad_w1, pad_w2, pad_h1, pad_h2 = get_pad(x)
        x = F.pad(x, [pad_w1, pad_w2, pad_h1, pad_h2], mode="reflect")
        out = _forward(model, x, enable_amp)
    else:
        x_org = x
        pad_w1, pad_w2, pad_h1, pad_h2 = get_pad(x)
        x = F.pad(x, [pad_w1, pad_w2, pad_h1, pad_h2], mode="reflect")
        out = _forward(model, x, enable_amp)
        if flip_aug:
            x = torch.flip(x_org, dims=[3])
            pad_w1, pad_w2, pad_h1, pad_h2 = get_pad(x)
            x = F.pad(x, [pad_w1, pad_w2, pad_h1, pad_h2], mode="reflect")
            out2 = _forward(model, x, enable_amp)
            out = torch.cat([out, out2], dim=0)

    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]),
                            mode="bicubic", align_corners=False)
    out = out[:, :, pad_h1:-pad_h2, pad_w1:-pad_w2]
    if flip_aug:
        if batch:
            n = out.shape[0] // 2
            z = torch.empty((n, *out.shape[1:]), device=out.device)
            for i in range(n):
                z[i] = (out[i] + torch.flip(out[i + n], dims=[2])) * 128
        else:
            z = (out[0:1] + torch.flip(out[1:2], dims=[3])) * 128
    else:
        z = out * 256
    if not batch:
        assert z.shape[0] == 1
        z = z.squeeze(0)

    z = z.to(output_device)
    if int16:
        z = z.to(torch.int16)

    return z


class HeightResizer():
    def __init__(self, h_height=384, v_height=512):
        self.h_height = h_height
        self.v_height = v_height

    def get_size(self, width, height):
        target_height = self.h_height if width > height else self.v_height
        if target_height < height:
            new_h = target_height
            new_w = int(new_h / height * width)
            if new_w % 32 != 0:
                new_w += (32 - new_w % 32)
            if new_h % 32 != 0:
                new_h += (32 - new_h % 32)
        else:
            new_h, new_w = height, width
            if new_w % 32 != 0:
                new_w -= new_w % 32
            if new_h % 32 != 0:
                new_h -= new_h % 32

        return new_w, new_h

    def __call__(self, x):
        width, height = x.shape[-2:][::-1]
        new_w, new_h = self.get_size(width, height)
        if new_w != width or new_h != height:
            x = F.interpolate(x, size=(new_h, new_w),
                              mode="bilinear", align_corners=True, antialias=False)
        return x


def _patch_resize_debug(model):
    resizer = model.core.prep.resizer
    if isinstance(resizer, HeightResizer):
        print("HeightResizer", resizer.v_height, resizer.h_height)
    else:
        print("Resizer",
              resizer._Resize__width,
              resizer._Resize__height,
              resizer._Resize__resize_method,
              resizer._Resize__keep_aspect_ratio,
              resizer._Resize__multiple_of)
    get_size = resizer.get_size

    def get_size_wrap(width, height):
        new_size = get_size(width, height)
        print("resize", (width, height), new_size)
        return new_size

    if resizer.get_size.__code__.co_code != get_size_wrap.__code__.co_code:
        resizer.get_size = get_size_wrap
