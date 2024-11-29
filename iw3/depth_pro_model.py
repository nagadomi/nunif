import os
from os import path
import gc
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps # noqa
from .dilation import dilate_edge
from . base_depth_model import BaseDepthModel, HUB_MODEL_DIR


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


def batch_preprocess(x, img_size=1536, padding=False):
    # x: BCHW float32 0-1
    B, C, H, W = x.shape

    def normalize(x):
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
        stdv = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
        x.sub_(mean).div_(stdv)
        return x

    antialias = False
    if not padding:
        x = normalize(x.clone())
        x = F.interpolate(x, size=(img_size, img_size),
                          mode="bilinear", align_corners=False, antialias=antialias)
        return x, 0
    else:
        pad = int(img_size * 0.25 ** 2)
        size = img_size - pad * 2
        x = normalize(x.clone())
        x = F.interpolate(x, size=(size, size),
                          mode="bilinear", align_corners=False, antialias=antialias)
        x = F.pad(x, (pad,) * 4, mode="reflect")

    return x, pad


def _forward(model, x, min_dist=0.1, max_dist=40.0):
    if x.dtype != torch.float16:
        x = x.half()
    H, W = x.shape[2:]
    canonical_inverse_depth, _ = model(x)
    canonical_inverse_depth = canonical_inverse_depth.to(torch.float32)

    if False:
        # distance
        inverse_depth = canonical_inverse_depth.mul_(0.6115)
        depth = 1.0 / torch.clamp(inverse_depth, min=1.0 / max_dist, max=1.0 / min_dist)
    else:
        # inverse distance
        # Fov=70mm fixed to avoid fov flicking
        inverse_depth = canonical_inverse_depth.mul_(0.6115)
        depth = torch.clamp(inverse_depth, max=1.0 / min_dist)
    return depth


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, enable_amp=False,
                output_device="cpu", device=None,
                edge_dilation=2,
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

    x, unpad = batch_preprocess(x, model.img_size)

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

    if unpad > 0:
        out = out[:, :, unpad:-unpad, unpad:-unpad]

    if edge_dilation > 0:
        out = dilate_edge(out, edge_dilation)
    out.neg_()

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

    z = z.to(output_device)

    return z


class DepthProModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        assert model_type in MODEL_FILES
        dtype = torch.float16
        encoder = NAME_MAP[model_type]
        if not os.getenv("IW3_DEBUG"):
            model, _ = torch.hub.load("nagadomi/ml-depth-pro_iw3:main",
                                      "DepthPro", img_size=encoder, device=device, dtype=dtype,
                                      verbose=False, trust_repo=True)
        else:
            assert path.exists("../ml-depth-pro_iw3/hubconf.py")
            model, _ = torch.hub.load("../ml-depth-pro_iw3",
                                      "DepthPro", img_size=encoder, device=device, dtype=dtype,
                                      source="local", verbose=False, trust_repo=True)

        model.device = device
        model.metric_depth = False

        # delete unused fov model
        delattr(model, "fov")

        # Release VRAM (there are many unused parameters that have not been released)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        return batch_infer(
            self.model, x, flip_aug=tta, low_vram=low_vram,
            enable_amp=enable_amp,
            output_device=x.device,
            device=x.device,
            edge_dilation=edge_dilation)

    @classmethod
    def get_name(cls):
        return "DepthPro"

    @classmethod
    def has_checkpoint_file(cls, model_type):
        return cls.supported(model_type) and path.exists(MODEL_FILES[model_type])

    @classmethod
    def supported(cls, model_type):
        return model_type in MODEL_FILES

    @classmethod
    def get_model_path(cls, model_type):
        return MODEL_FILES[model_type]

    def is_metric(self):
        return False  # use inverse_depth

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return False  # TODO: not tested

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/ml-depth-pro_iw3:main", "DepthPro")

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


def _bench():
    import time

    N = 10
    model = DepthProModel("DepthPro")
    model.load(gpu=0)
    x = torch.randn((1, 3, 1536, 1536)).cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        t = time.time()
        for _ in range(N):
            _forward(model.get_model(), x)
        torch.cuda.synchronize()
        print(round((time.time() - t) / N, 4))


def _test():
    from PIL import Image
    import cv2
    import numpy as np

    model = DepthProModel("DepthPro")
    model.load(gpu=0)
    im = Image.open("cc0/dog2.jpg").convert("RGB")
    out = batch_infer(model.get_model(), im, flip_aug=False, int16=True,
                      enable_amp=True, output_device="cpu", device="cuda")
    out = out.squeeze(0).numpy().astype(np.uint16)
    cv2.imwrite("./tmp/depth_pro_out.png", out)


if __name__ == "__main__":
    _test()
    # _bench()
