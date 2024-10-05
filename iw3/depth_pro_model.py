import os
from os import path
import pickle
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.device import create_device, autocast, device_is_mps # noqa
from nunif.models.data_parallel import DeviceSwitchInference
from .dilation import dilate_edge


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
NAME_MAP = {
    "DepthPro": 384,
    "DepthPro_HD": 256,
    "DepthPro_SD": 128,
}
MODEL_FILES = {
    "DepthPro": path.join(HUB_MODEL_DIR, "checkpoints", "depth_pro.pt"),
    "DepthPro_HD": path.join(HUB_MODEL_DIR, "checkpoints", "depth_pro.pt"),
    "DepthPro_SD": path.join(HUB_MODEL_DIR, "checkpoints", "depth_pro.pt"),
}


def get_name():
    return "DepthPro"


def load_model(model_type="DepthPro", gpu=0, **kwargs):
    assert model_type in MODEL_FILES
    device = create_device(gpu)

    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        try:
            encoder = NAME_MAP[model_type]
            if not os.getenv("IW3_DEBUG"):
                model, _ = torch.hub.load("nagadomi/ml-depth-pro_iw3:main",
                                          "DepthPro", img_size=encoder, device=device, dtype=torch.float16,
                                          verbose=False, trust_repo=True)
            else:
                assert path.exists("../ml-depth-pro_iw3/hubconf.py")
                model, _ = torch.hub.load("../ml-depth-pro_iw3",
                                          "DepthPro", img_size=encoder, device=device, dtype=torch.float16,
                                          source="local", verbose=False, trust_repo=True)
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

    model.device = device
    model.metric_depth = True
    if isinstance(gpu, (list, tuple)) and len(gpu) > 1:
        model = DeviceSwitchInference(model, device_ids=gpu)

    return model


def has_model(model_type):
    assert model_type in MODEL_FILES
    return path.exists(MODEL_FILES[model_type])


def force_update():
    with TorchHubDir(HUB_MODEL_DIR):
        torch.hub.help("nagadomi/ml-depth-pro_iw3:main", "DepthPro",
                       force_reload=True, trust_repo=True)


def batch_preprocess(x, img_size=1536):
    # x: BCHW float32 0-1
    B, C, H, W = x.shape
    antialias = True
    x = F.interpolate(x, size=(img_size, img_size),
                      mode="bilinear", align_corners=False, antialias=antialias)
    x.clamp_(0, 1)

    # normalize
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    x.sub_(mean).div_(stdv)
    return x


def _forward(model, x, min_dist=0.01, max_dist=40.0):
    if x.dtype != torch.float16:
        x = x.half()
    H, W = x.shape[2:]
    canonical_inverse_depth, fov_deg = model(x)
    canonical_inverse_depth = canonical_inverse_depth.to(torch.float32)

    # TODO: weird value range
    if True:
        fov_deg = fov_deg.to(torch.float32)
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg))
        inverse_depth = canonical_inverse_depth * (W / f_px)
        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        depth = torch.clamp(depth, min=min_dist, max=max_dist)
    else:
        depth = -canonical_inverse_depth

    return depth


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
    x = batch_preprocess(x, model.img_size)

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
        out = -dilate_edge(-out, edge_dilation)
    if resize_depth and out.shape[-2:] != org_size:
        out = F.interpolate(out, size=(org_size[0], org_size[1]),
                            mode="bilinear", align_corners=False, antialias=True)
    if flip_aug:
        if batch:
            n = out.shape[0] // 2
            z = torch.empty((n, *out.shape[1:]), device=out.device)
            for i in range(n):
                z[i] = (out[i] + torch.flip(out[i + n], dims=[2])) * 0.5
        else:
            z = (out[0:1] + torch.flip(out[1:2], dims=[3])) * 0.5
    else:
        z = out
    if not batch:
        assert z.shape[0] == 1
        z = z.squeeze(0)

    if int16:
        if normalize_int16:
            max_v, min_v = z.max(), z.min()
            print(max_v, min_v, z.mean())
            uint16_max = 0xffff
            if max_v - min_v > 0:
                z = uint16_max * ((z - min_v) / (max_v - min_v))
            else:
                z = torch.zeros_like(z)
        z = z.to(torch.int16)

    z = z.to(output_device)

    return z


def _bench():
    import time

    N = 10
    model = load_model(model_type="DepthPro", gpu=0)
    x = torch.randn((1, 3, 1536, 1536)).cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        t = time.time()
        for _ in range(N):
            _forward(model, x)
        torch.cuda.synchronize()
        print(round((time.time() - t) / N, 4))


def _test():
    from PIL import Image
    import cv2
    import numpy as np

    model = load_model(model_type="DepthPro_HD")
    im = Image.open("cc0/dog2.jpg").convert("RGB")
    out = batch_infer(model, im, flip_aug=False, int16=True,
                      enable_amp=True, output_device="cpu", device="cuda")
    out = out.squeeze(0).numpy().astype(np.uint16)
    cv2.imwrite("./tmp/depth_pro_out.png", out)


if __name__ == "__main__":
    # _test()
    _bench()
