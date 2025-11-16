import os
from os import path
import torch
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps, device_is_xpu # noqa
from .dilation import dilate_edge
from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR
from .depth_anything_model import batch_preprocess
from .models import DepthAA


NAME_MAP = {
    "Any_V3_Mono": "da3mono-large",
}
MODEL_FILES = {
    "Any_V3_Mono": path.join(HUB_MODEL_DIR, "checkpoints", "da3mono-large.safetensors"),
}
AA_SUPPORTED_MODELS = {
    "Any_V3_Mono",
}


def _forward(model, x, enable_amp):
    with autocast(device=x.device, enabled=enable_amp):
        x = x.unsqueeze(1)  # (B, S, C, H, W)
        out = model(x)

    depths = out["depth"].float()
    sky_masks = out["sky"] > 0.3

    disparity_maps = []
    for depth, sky_mask in zip(depths, sky_masks):
        # TODO: improve this
        median_rel_dist = torch.median(depth)
        depth = depth / (median_rel_dist + 1e-6)
        max_rel_dist = torch.quantile(depth, 0.98)
        depth = torch.where(torch.logical_or(sky_mask, depth > max_rel_dist), max_rel_dist, depth)
        depth = 1.0 / (depth + 0.1)
        disparity_maps.append(depth)

    out = torch.stack(disparity_maps)
    out = torch.nan_to_num(out)
    return out


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, enable_amp=False,
                output_device="cpu", device=None, edge_dilation=2, depth_aa=None,
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
    if depth_aa is not None:
        out = depth_aa.infer(out)

    if edge_dilation > 0:
        out = dilate_edge(out, edge_dilation)

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


class DepthAnythingV3MonoModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        # load aa model
        if model_type in AA_SUPPORTED_MODELS:
            self.depth_aa = DepthAA().load().eval().to(device)
        else:
            self.depth_aa = None

        model_name = NAME_MAP[model_type]
        if not os.getenv("IW3_DEBUG"):
            model = torch.hub.load("nagadomi/Depth-Anything-3_iw3:main",
                                   "load_model", model_name=model_name,
                                   verbose=False, trust_repo=True)
        else:
            assert path.exists("../Depth-Anything-3_iw3/hubconf.py")
            model = torch.hub.load("../Depth-Anything-3_iw3",
                                   "load_model", model_name=model_name, source="local",
                                   verbose=False, trust_repo=True)

        model.prep_lower_bound = resolution or 392
        if model.prep_lower_bound % 14 != 0:
            # From GUI, 512 -> 504
            model.prep_lower_bound -= model.prep_lower_bound % 14
        model.device = device

        return model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False, **kwargs):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        return batch_infer(
            self.model, x, flip_aug=tta, low_vram=low_vram,
            enable_amp=enable_amp,
            output_device=x.device,
            device=x.device,
            edge_dilation=edge_dilation,
            depth_aa=self.depth_aa if depth_aa else None,
        )

    @classmethod
    def get_name(cls):
        return "DepthAnythingV3Mono"

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
        return False

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/Depth-Anything-3_iw3:main", "load_model")

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


def _bench():
    import time

    B = 4
    N = 100
    model = DepthAnythingV3MonoModel("Any_V3_MONO")
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


if __name__ == "__main__":
    _bench()
