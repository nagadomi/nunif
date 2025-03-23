import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps, device_is_xpu # noqa
from .dilation import dilate_edge
from . base_depth_model import BaseDepthModel, HUB_MODEL_DIR


NAME_MAP = {
    "Any_S": "vits",
    "Any_B": "vitb",
    "Any_L": "vitl",
    "Any_V2_S": "v2_vits",
    "Any_V2_B": "v2_vitb",
    "Any_V2_L": "v2_vitl",

    "Any_V2_N_S": "hypersim_s",
    "Any_V2_N_B": "hypersim_b",
    "Any_V2_N_L": "hypersim_l",
    "Any_V2_K_S": "vkitti_s",
    "Any_V2_K_B": "vkitti_b",
    "Any_V2_K_L": "vkitti_l",

    # for compatibility
    "Any_V2_N": "hypersim_l",
    "Any_V2_K": "vkitti_l",

    # Distill Any Depth
    "Distill_Any_S": "distill_any_depth_s",
    "Distill_Any_B": "distill_any_depth_b",
    "Distill_Any_L": "distill_any_depth_l",
}
MODEL_FILES = {
    "Any_S": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vits14.pth"),
    "Any_B": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vitb14.pth"),
    "Any_L": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_vitl14.pth"),
    "Any_V2_S": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_vits.pth"),
    "Any_V2_B": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_vitb.pth"),
    "Any_V2_L": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_vitl.pth"),

    "Any_V2_N_S": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth"),
    "Any_V2_N_B": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vitb.pth"),
    "Any_V2_N_L": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vitl.pth"),

    "Any_V2_K_S": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_vkitti_vits.pth"),
    "Any_V2_K_B": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_vkitti_vitb.pth"),
    "Any_V2_K_L": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_vkitti_vitl.pth"),

    # for compatibility
    "Any_V2_N": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vitl.pth"),
    "Any_V2_K": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_v2_metric_vkitti_vitl.pth"),

    # Distill Any Depth
    "Distill_Any_S": path.join(HUB_MODEL_DIR, "checkpoints", "distill_any_depth_vits.safetensors"),
    "Distill_Any_B": path.join(HUB_MODEL_DIR, "checkpoints", "distill_any_depth_vitb.safetensors"),
    "Distill_Any_L": path.join(HUB_MODEL_DIR, "checkpoints", "distill_any_depth_vitl.safetensors"),
}


def batch_preprocess(x, lower_bound=392, max_aspect_ratio=4):
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

    # Limit aspect ratio to avoid OOM
    if new_h < new_w:
        new_w = min(new_w, int(max_aspect_ratio * new_h))
    else:
        new_h = min(new_h, int(max_aspect_ratio * new_w))

    new_h -= new_h % ensure_multiple_of
    new_w -= new_w % ensure_multiple_of
    if new_h < lower_bound:
        new_h = lower_bound
    if new_w < lower_bound:
        new_w = lower_bound

    # TODO: 'aten::_upsample_bilinear2d_aa.out' is not currently implemented for mps/xpu device
    antialias = not (device_is_mps(x.device) or device_is_xpu(x.device))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=antialias)
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
    out = torch.nan_to_num(out)
    return out


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, enable_amp=False,
                output_device="cpu", device=None, edge_dilation=2,
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
        if not model.metric_depth:
            out = dilate_edge(out, edge_dilation)
        else:
            out = dilate_edge(out.neg_(), edge_dilation).neg_()

    if not model.metric_depth:
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

    z = z.to(output_device)

    return z


class DepthAnythingModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        encoder = NAME_MAP[model_type]
        if encoder.startswith("hypersim") or encoder.startswith("vkitti"):
            # Depth-Anything V2 metric depth model
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                       "DepthAnythingMetricDepthV2", model_type=encoder,
                                       verbose=False, trust_repo=True)
            else:
                assert path.exists("../Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load("../Depth-Anything_iw3",
                                       "DepthAnythingMetricDepthV2", model_type=encoder, source="local",
                                       verbose=False, trust_repo=True)
        elif encoder.startswith("distill_any_depth"):
            # Distill Any Depth
            encoder = {"l": "v2_vitl", "b": "v2_vitb", "s": "v2_vits"}[encoder[-1]]
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                       "DistillAnyDepth", encoder=encoder,
                                       verbose=False, trust_repo=True)
            else:
                assert path.exists("../Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load("../Depth-Anything_iw3",
                                       "DistillAnyDepth", encoder=encoder, source="local",
                                       verbose=False, trust_repo=True)
        else:
            # DepthAnything V1 or V2
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                       "DepthAnything", encoder=encoder,
                                       verbose=False, trust_repo=True)
            else:
                assert path.exists("../Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load("../Depth-Anything_iw3",
                                       "DepthAnything", encoder=encoder, source="local",
                                       verbose=False, trust_repo=True)

        model.metric_depth = getattr(model, "metric_depth", False)
        model.prep_lower_bound = resolution or 392
        if model.prep_lower_bound % 14 != 0:
            # From GUI, 512 -> 518
            model.prep_lower_bound += (14 - model.prep_lower_bound % 14)
        model.device = device

        return model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        return batch_infer(
            self.model, x, flip_aug=tta, low_vram=low_vram,
            enable_amp=enable_amp,
            output_device=x.device,
            device=x.device,
            edge_dilation=edge_dilation,
            resize_depth=False)

    @classmethod
    def get_name(cls):
        return "DepthAnything"

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
        if self.model is not None:
            return self.model.metric_depth
        else:
            return self.model_type.startswith("Any_V2_N") or self.model_type.startswith("Any_V2_K")

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/Depth-Anything_iw3:main", "DepthAnything")

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


def _bench():
    import time

    B = 4
    N = 100
    model = DepthAnythingModel("Any_L")
    model.load(gpu=0)
    x = torch.randn((B, 3, 392, 392)).cuda()
    model.infer(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        t = time.time()
        for _ in range(N):
            model.infer(x)
        torch.cuda.synchronize()
        print(round(1.0 / ((time.time() - t) / (B * N)), 4), "FPS")


def _test():
    from PIL import Image
    import cv2
    import numpy as np

    model = DepthAnythingModel("Any_S")
    model.load()
    im = Image.open("waifu2x/docs/images/miku_128.png").convert("RGB")
    out = model.infer_raw(im, flip_aug=False, int16=True,
                          enable_amp=True, output_device="cpu", device="cuda")
    out = out.squeeze(0).numpy().astype(np.uint16)
    cv2.imwrite("./tmp/depth_anything_out.png", out)


def _test_model():
    DepthAnythingModel()


if __name__ == "__main__":
    # _test()
    _bench()
    # _test_model()
