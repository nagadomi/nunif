import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps, device_is_xpu # noqa
from nunif.models.utils import compile_model
from nunif.modules.reflection_pad2d import reflection_pad2d_naive
from .dilation import dilate_edge
from . base_depth_model import BaseDepthModel, HUB_MODEL_DIR
from . depth_anything_model import batch_preprocess as batch_preprocess_da
from .models import DepthAA


NAME_MAP = {
    "VDA_Stream_S": "vits",
    "VDA_Stream_B": "vitb",
    "VDA_Stream_L": "vitl",
    "VDA_Stream_Metric_S": "vits",
    "VDA_Stream_Metric_B": "vitb",
    "VDA_Stream_Metric_L": "vitl",
}
MODEL_FILES = {
    "VDA_Stream_S": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vits.pth"),
    "VDA_Stream_B": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vitb.pth"),
    "VDA_Stream_L": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vitl.pth"),
    "VDA_Stream_Metric_S": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vits.pth"),
    "VDA_Stream_Metric_B": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vitb.pth"),
    "VDA_Stream_Metric_L": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vitl.pth"),
}
AA_SUPPORT_MODELS = {
    "VDA_Stream_S",
    "VDA_Stream_B",
    "VDA_Stream_L",
    "VDA_Stream_Metric_S",
    "VDA_Stream_Metric_B",
    "VDA_Stream_Metric_L",
}
METRIC_PADDING = 14
METRIC_DEPTH_TYPES = {
    "VDA_Stream_Metric_S",
    "VDA_Stream_Metric_B",
    "VDA_Stream_Metric_L",
}


def batch_preprocess(x, lower_bound, metric_depth):
    if metric_depth:
        x = batch_preprocess_da(x, lower_bound - METRIC_PADDING * 2)
        x = reflection_pad2d_naive(x, (METRIC_PADDING,) * 4)
    else:
        x = batch_preprocess_da(x, lower_bound)
    assert x.shape[2] % 14 == 0 and x.shape[3] % 14 == 0
    return x


def postprocess(out, edge_dilation, metric_depth, max_dist=None, depth_aa=None, enable_amp=True):
    out = torch.nan_to_num(out)
    print(out.min(), out.max())

    if max_dist is not None:
        out = torch.clamp(out, max=max_dist)

    if depth_aa is not None:
        ori_dtype = out.dtype
        with autocast(device=out.device, enabled=enable_amp):
            out = depth_aa.infer(out)
        out = out.to(ori_dtype)

    if metric_depth:
        out = F.pad(out, (-METRIC_PADDING,) * 4)

    if edge_dilation > 0:
        out = dilate_edge(out, edge_dilation)

    # invert for zoedepth compatibility
    if not metric_depth:
        # invert for zoedepth compatibility
        out.neg_()
    if out.dtype != torch.float32:
        out = out.to(torch.float32)

    return out


class VideoDepthAnythingStreamingModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        # load aa model
        self.depth_aa = DepthAA().load().eval().to(device)
        # load depth model
        encoder = NAME_MAP[model_type]
        metric_depth = model_type in METRIC_DEPTH_TYPES
        if not os.getenv("IW3_DEBUG"):
            model = torch.hub.load("nagadomi/Video-Depth-Anything_iw3:main",
                                   "VideoDepthAnythingStreaming", encoder=encoder, metric_depth=metric_depth, device=device,
                                   verbose=False, trust_repo=True)
        else:
            assert path.exists("../Video-Depth-Anything_iw3/hubconf.py")
            model = torch.hub.load("../Video-Depth-Anything_iw3",
                                   "VideoDepthAnythingStreaming", encoder=encoder, metric_depth=metric_depth, device=device,
                                   source="local", verbose=False, trust_repo=True)

        model.prep_lower_bound = resolution or 392
        if model.prep_lower_bound % 14 != 0:
            # From GUI, 512 -> 518
            model.prep_lower_bound += (14 - model.prep_lower_bound % 14)

        return model

    def reset_state(self):
        self.model.reset_state()

    def infer(self, x, enable_amp=True, edge_dilation=0, depth_aa=False, **kwargs):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)

        depth_aa = self.depth_aa if depth_aa else None

        batch = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
        else:
            batch = True

        x = batch_preprocess(x, self.model.prep_lower_bound, metric_depth=self.is_metric())
        outputs = []
        for frame in x:
            outputs.append(self.model.infer_video_depth_one(frame, use_amp=enable_amp).to(torch.float32))
        depth = torch.stack(outputs)
        depth = postprocess(depth, edge_dilation=edge_dilation, depth_aa=depth_aa, metric_depth=self.is_metric(),
                            enable_amp=enable_amp)
        if not batch:
            depth = depth.squeeze(0)

        return depth

    @classmethod
    def get_name(cls):
        return "VideoDepthAnythingStreaming"

    def is_image_supported(self):
        return False

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
        return self.model_type in METRIC_DEPTH_TYPES

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return False  # TODO

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/Video-Depth-Anything_iw3:main", "VideoDepthAnythingStreaming")

    def compile(self):
        if self.model_backup is None:
            self.model_backup = (self.model.head, self.model.pretrained)
            self.model.head = compile_model(self.model.head)
            self.model.pretrained = compile_model(self.model.pretrained)

    def clear_compiled_model(self):
        if self.model_backup is not None:
            self.model.head, self.model.pretrained = self.model_backup
            self.model_backup = None


def _test():
    from PIL import Image
    import torchvision.transforms.functional as TF

    model = VideoDepthAnythingStreamingModel("VDA_Stream_S")
    model.load(gpu=0)
    im = Image.open("cc0/320/dog.png").convert("RGB")
    x = TF.to_tensor(im).to(model.device)
    out = model.infer(x)
    print(out.shape)


if __name__ == "__main__":
    _test()
