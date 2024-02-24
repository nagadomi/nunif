import os
from os import path
import pickle
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.device import create_device, autocast
from nunif.models.data_parallel import DataParallelInference
from .dilation import dilate_edge


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
NAME_MAP = {
    "Any_S": "vits",
    "Any_B": "vitb",
    "Any_L": "vitl"
}
MODEL_FILES = {
    "Any_S": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vits14.pth"),
    "Any_B": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vitb14.pth"),
    "Any_L": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vitl14.pth"),
}


def get_name():
    return "DepthAnything"


def load_model(model_type="Any_B", gpu=0, **kwargs):
    assert model_type in MODEL_FILES
    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        try:
            encoder = NAME_MAP[model_type]
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                       "DepthAnything", encoder=encoder,
                                       verbose=False, trust_repo=True)
            else:
                assert path.exists("../Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load("../Depth-Anything_iw3",
                                       "DepthAnything", encoder=encoder, source="local",
                                       verbose=False, trust_repo=True)
        except (RuntimeError, pickle.PickleError) as e:
            if isinstance(e, RuntimeError):
                do_handle = "PytorchStreamReader" in repr(e)
            else:
                do_handle = True
            if do_handle:
                try:
                    # delete corrupted file
                    os.unlink(MODEL_FILES[model_type])
                except:  # noqa
                    pass
                raise RuntimeError(
                    f"File `{MODEL_FILES[model_type]}` is corrupted. "
                    "This error may occur when the network is unstable or the disk is full. "
                    "Try again."
                )
            else:
                raise

    device = create_device(gpu)
    model = model.to(device).eval()
    model.device = device
    model.prep_lower_bound = kwargs.get("height", None) or 392
    if model.prep_lower_bound % 14 != 0:
        # From GUI, 512 -> 518
        model.prep_lower_bound += (14 - model.prep_lower_bound % 14)

    if isinstance(gpu, (list, tuple)) and len(gpu) > 1:
        model = DataParallelInference(model, device_ids=gpu)

    return model


def has_model(model_type):
    assert model_type in MODEL_FILES
    return path.exists(MODEL_FILES[model_type])


def force_update():
    with TorchHubDir(HUB_MODEL_DIR):
        torch.hub.help("nagadomi/Depth-Anything_iw3:main", "DepthAnything",
                       force_reload=True, trust_repo=True)


def batch_preprocess(x, lower_bound=392):
    # x: BCHW float32 0-1
    B, C, H, W = x.shape

    # resize
    ensure_multiple_of = 14
    if W < H:
        scale_factor = lower_bound / W
    else:
        scale_factor = lower_bound / H
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)
    new_h -= new_h % ensure_multiple_of
    new_w -= new_w % ensure_multiple_of
    if new_h < lower_bound:
        new_h = lower_bound
    if new_w < lower_bound:
        new_w = lower_bound
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    x.clamp_(0, 1)

    # normalize
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    x.sub_(mean).div_(stdv)
    return x


def _forward(model, x, enable_amp):
    with autocast(device=x.device, enabled=enable_amp):
        out = model(x).unsqueeze(dim=1)
    if out.dtype != torch.float32:
        out = out.to(torch.float32)
    return out


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, int16=True, enable_amp=False,
                output_device="cpu", device=None, normalize_int16=True,
                edge_dilation=2, resize_depth=True,
                **kwargs):
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

    org_size = x.shape[-2:]
    x = batch_preprocess(x, model.prep_lower_bound)

    if not low_vram:
        if flip_aug:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        out = _forward(model, x, enable_amp)
    else:
        x_org = x
        out = _forward(model, x, enable_amp)
        if flip_aug:
            x = torch.flip(x_org, dims=[3])
            out2 = _forward(model, x, enable_amp)
            out = torch.cat([out, out2], dim=0)

    if edge_dilation > 0:
        out = dilate_edge(out, edge_dilation)
    if resize_depth and out.shape[-2:] != org_size:
        out = F.interpolate(out, size=(org_size[0], org_size[1]),
                            mode="bilinear", align_corners=False, antialias=True)

    # invert for zoedepth compatibility
    out.neg_()

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

    if int16:
        # 1. ignore normalize_int16
        # 2. DepthAnything output is relative depth so normalize
        # TODO: torch.uint16 will be implemented.
        #       For now, use numpy to save uint16 png.
        #  torch.tensor([0, 1, 65534], dtype=torch.float32).to(torch.int16).numpy().astype(np.uint16)
        # >array([    0,     1, 65534], dtype=uint16)
        max_v, min_v = z.max(), z.min()
        uint16_max = 0xffff
        if max_v - min_v > 0:
            z = uint16_max * ((z - min_v) / (max_v - min_v))
        else:
            z = torch.zeros_like(z)
        z = z.to(torch.int16)

    z = z.to(output_device)

    return z


def _bench():
    from PIL import Image
    import cv2
    import numpy as np
    import time

    N = 100

    model = load_model(model_type="Any_L", gpu=0)
    x = torch.randn((1, 3, 518, 784)).cuda()
    with torch.no_grad():
        t = time.time()
        for _ in range(N):
            z = model(x)
            torch.cuda.synchronize()
        print(round((time.time() - t) / N, 4))


def _test():
    from PIL import Image
    import cv2
    import numpy as np

    model = load_model()
    im = Image.open("waifu2x/docs/images/miku_128.png").convert("RGB")
    out = batch_infer(model, im, flip_aug=False, int16=True,
                      enable_amp=True, output_device="cpu", device="cuda")
    out = out.squeeze(0).numpy().astype(np.uint16)
    cv2.imwrite("./tmp/depth_anything_out.png", out)


if __name__ == "__main__":
    #_test()
    _bench()
