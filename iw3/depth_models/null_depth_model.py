"""Dummy depth model for performance benchmark"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from ..base_depth_model import BaseDepthModel
from ..dilation import dilate_edge, edge_dilation_is_enabled


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
    def infer(
        self,
        x: torch.Tensor | Image.Image,
        tta: bool = False,
        low_vram: bool = False,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_tensor: torch.Tensor = x
        else:
            x_tensor = TF.to_tensor(x)
        batch = True
        if x_tensor.ndim == 3:
            batch = False
            x_tensor = x_tensor.unsqueeze(0)

        if tta:
            x_out = (self.model(x_tensor) + self.model(x_tensor)) * 0.5
        else:
            x_out = self.model(x_tensor)

        if edge_dilation_is_enabled(edge_dilation):
            x_out = dilate_edge(x_out, edge_dilation)

        if not batch:
            x_out = x_out.squeeze(0)

        return x_out

    order: int = 999

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
    def get_model_types(cls) -> list[str]:
        return ["NULL"]

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
