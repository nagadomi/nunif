""" Dummy depth model for performance benchmark
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilation import dilate_edge
from . base_depth_model import BaseDepthModel


class NullDepth(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        if resolution is None:
            resolution = 392
        self.resolution = resolution

    def forward(self, x):
        # resize + grayscale only
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear")
        x = x.mean(dim=1, keepdim=True)
        return x


class NullDepthModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, **kwargs):
        return NullDepth(resolution)

    @torch.inference_mode()
    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, **kwargs):
        batch = True
        if x.ndim == 3:
            batch = False
            x = x.unsqueeze(0)

        if tta:
            x = (self.model(x) + self.model(x)) * 0.5
        else:
            x = self.model(x)
        if edge_dilation > 0:
            x = dilate_edge(x, edge_dilation)

        if not batch:
            x = x.squeeze(0)

        return x

    @classmethod
    def get_name(cls):
        return "NullDepth"

    @classmethod
    def has_checkpoint_file(cls, model_type):
        return cls.supported(model_type)

    @classmethod
    def supported(cls, model_type):
        return model_type == "NULL"

    @classmethod
    def get_model_path(cls, model_type):
        return None

    def is_metric(self):
        return False

    def is_video_supported(self):
        return True

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        pass
