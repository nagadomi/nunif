import os
from os import path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from nunif.device import autocast, create_device, device_is_mps, device_is_xpu  # noqa
from nunif.models.utils import compile_function, compile_model
from nunif.modules.reflection_pad2d import reflection_pad2d_naive

from ..base_depth_model import HUB_MODEL_DIR, BaseDepthModel
from ..dilation import dilate_edge, edge_dilation_is_enabled
from ..models import DepthAA
from .depth_anything_model import batch_preprocess as batch_preprocess_da

NAME_MAP = {
    "VDA_S": "vits",
    "VDA_B": "vitb",
    "VDA_L": "vitl",
    "VDA_Metric_S": "vits",
    "VDA_Metric_B": "vitb",
    "VDA_Metric_L": "vitl",
}
MODEL_FILES = {
    "VDA_S": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vits.pth"),
    "VDA_B": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vitb.pth"),
    "VDA_L": path.join(HUB_MODEL_DIR, "checkpoints", "video_depth_anything_vitl.pth"),
    "VDA_Metric_S": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vits.pth"),
    "VDA_Metric_B": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vitb.pth"),
    "VDA_Metric_L": path.join(HUB_MODEL_DIR, "checkpoints", "metric_video_depth_anything_vitl.pth"),
}
METRIC_PADDING = 14
AA_SUPPORT_MODELS = {
    "VDA_S",
    "VDA_B",
    "VDA_L",
    "VDA_Metric_S",
    "VDA_Metric_B",
    "VDA_Metric_L",
}
METRIC_DEPTH_TYPES = {
    "VDA_Metric_S",
    "VDA_Metric_B",
    "VDA_Metric_L",
}


def batch_preprocess(x, lower_bound, metric_depth, limit_resolution=False):
    if metric_depth:
        x = batch_preprocess_da(x, lower_bound - METRIC_PADDING * 2, limit_resolution=limit_resolution)
        x = reflection_pad2d_naive(x, (METRIC_PADDING,) * 4)
    else:
        x = batch_preprocess_da(x, lower_bound, limit_resolution=limit_resolution)
    assert x.shape[2] % 14 == 0 and x.shape[3] % 14 == 0
    return x


def _postprocess(
    out, edge_dilation, metric_depth, force_disparity=False, max_dist=None, depth_aa=None, enable_amp=True
):
    out = out.unsqueeze(1)
    out = torch.nan_to_num(out)

    if max_dist is not None:
        out = torch.clamp(out, max=max_dist)

    if metric_depth and force_disparity:
        # convert to disparity
        eps = 0.1
        out = 1.0 / (out + eps)

    if depth_aa is not None:
        ori_dtype = out.dtype
        with autocast(device=out.device, enabled=enable_amp):
            out = depth_aa.infer(out)
        out = out.to(ori_dtype)

    if metric_depth:
        out = F.pad(out, (-METRIC_PADDING,) * 4)

    is_disparity = not metric_depth or (metric_depth and force_disparity)
    if edge_dilation_is_enabled(edge_dilation):
        if is_disparity:
            out = dilate_edge(out, edge_dilation)
        else:
            out = -dilate_edge(-out, edge_dilation)

    if not is_disparity:
        # invert for zoedepth compatibility
        out = -out
    if out.dtype != torch.float32:
        out = out.to(torch.float32)
    return out


def postprocess(out, edge_dilation, metric_depth, max_dist=None, depth_aa=None, force_disparity=False, enable_amp=True):
    micro_batch_size = 4
    return torch.cat(
        [
            _postprocess(
                batch,
                edge_dilation=edge_dilation,
                metric_depth=metric_depth,
                force_disparity=force_disparity,
                max_dist=max_dist,
                depth_aa=depth_aa,
                enable_amp=enable_amp,
            )
            for batch in torch.split(out, micro_batch_size, dim=0)
        ],
        dim=0,
    )


class VideoDepthAnythingModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.input_frame_count = 0
        self.output_frame_count = 0
        self.metric_depth = model_type in METRIC_DEPTH_TYPES
        # if True, use 1 / depth and is_metric==False
        self.force_disparity = True

    def load_model(self, model_type, resolution=None, device=None):
        # load aa model
        self.depth_aa = DepthAA().load().eval().to(device)
        # load depth model
        encoder = NAME_MAP[model_type]
        if self.metric_depth:
            # MetricVideoDepthAnything
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load(
                    "nagadomi/Video-Depth-Anything_iw3:main",
                    "MetricVideoDepthAnythingOnline",
                    encoder=encoder,
                    device=device,
                    verbose=False,
                    trust_repo=True,
                )
            else:
                assert path.exists("../Video-Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load(
                    "../Video-Depth-Anything_iw3",
                    "MetricVideoDepthAnythingOnline",
                    encoder=encoder,
                    device=device,
                    source="local",
                    verbose=False,
                    trust_repo=True,
                )
        else:
            # VideoDepthAnything
            if not os.getenv("IW3_DEBUG"):
                model = torch.hub.load(
                    "nagadomi/Video-Depth-Anything_iw3:main",
                    "VideoDepthAnythingOnline",
                    encoder=encoder,
                    device=device,
                    verbose=False,
                    trust_repo=True,
                )
            else:
                assert path.exists("../Video-Depth-Anything_iw3/hubconf.py")
                model = torch.hub.load(
                    "../Video-Depth-Anything_iw3",
                    "VideoDepthAnythingOnline",
                    encoder=encoder,
                    device=device,
                    source="local",
                    verbose=False,
                    trust_repo=True,
                )

        model.prep_lower_bound = resolution or 392
        if model.prep_lower_bound % 14 != 0:
            # From GUI, 512 -> 518
            model.prep_lower_bound += 14 - model.prep_lower_bound % 14

        return model

    def reset_state(self):
        self.model.reset_state()
        self.input_frame_count = 0
        self.output_frame_count = 0

    def reset(self):
        self.reset_state()
        self.reset_ema()

    def infer(
        self,
        x: torch.Tensor | Image.Image,
        tta: bool = False,
        low_vram: bool = False,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        streaming: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).unsqueeze(0).to(self.device)
        assert torch.is_tensor(x)

        if not streaming and x.shape[0] == 1:
            # Standalone single image inference (tests, etc.)
            self.reset_state()
            x_in = batch_preprocess(
                x,
                self.model.prep_lower_bound,
                metric_depth=self.model.metric_depth,
                limit_resolution=self.limit_resolution,
            )
            self.model.infer(x_in[0], use_amp=enable_amp)
            self.input_frame_count = 1
            ret = self._flush(enable_amp=enable_amp)
            if ret:
                out = torch.stack(ret)
                depth_aa_module = self.depth_aa if depth_aa else None
                out = postprocess(
                    out,
                    edge_dilation=edge_dilation,
                    depth_aa=depth_aa_module,
                    metric_depth=self.model.metric_depth,
                    force_disparity=self.force_disparity,
                    enable_amp=enable_amp,
                )
                self.reset_state()
                return out.squeeze(0) if x.ndim == 3 else out
            self.reset_state()
            return None

        # Streaming video inference (x is BCHW or CHW)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x_in = batch_preprocess(
            x,
            self.model.prep_lower_bound,
            metric_depth=self.model.metric_depth,
            limit_resolution=self.limit_resolution,
        )
        depth_aa_module = self.depth_aa if depth_aa else None
        ret_all = []
        for i in range(B):
            self.input_frame_count += 1
            ret = self.model.infer(x_in[i], use_amp=enable_amp)
            if ret is not None:
                self.output_frame_count += len(ret)
                ret_all += ret

        if ret_all:
            out = torch.stack(ret_all)
            return postprocess(
                out,
                edge_dilation=edge_dilation,
                depth_aa=depth_aa_module,
                metric_depth=self.model.metric_depth,
                force_disparity=self.force_disparity,
                enable_amp=enable_amp,
            )
        return None

    def flush(
        self,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        ret = self._flush(enable_amp=enable_amp)
        if ret:
            out = torch.stack(ret)
            depth_aa_module = self.depth_aa if depth_aa else None
            return postprocess(
                out,
                edge_dilation=edge_dilation,
                depth_aa=depth_aa_module,
                metric_depth=self.model.metric_depth,
                force_disparity=self.force_disparity,
                enable_amp=enable_amp,
            )
        return None

    def infer_with_normalize(self, x, pts, reset_pts, enable_amp=True, edge_dilation=0, depth_aa=None, **kwargs):
        assert x.ndim == 4
        depth_aa_bool = bool(depth_aa)
        B = x.shape[0]
        outputs = []
        for i in range(B):
            out = self.infer(
                x[i : i + 1],
                enable_amp=enable_amp,
                edge_dilation=edge_dilation,
                depth_aa=depth_aa_bool,
                streaming=True,
            )
            if out is not None:
                for j in range(out.shape[0]):
                    normalized_depth = self.minmax_normalize_chw(out[j])
                    if normalized_depth is not None:
                        outputs.append(normalized_depth)
            if pts[i] in reset_pts:
                outputs += self.flush_with_normalize(
                    enable_amp=enable_amp, edge_dilation=edge_dilation, depth_aa=depth_aa_bool
                )
                self.reset()
        return outputs

    def flush_with_normalize(self, enable_amp=True, edge_dilation=0, depth_aa=None):
        depth_aa_bool = bool(depth_aa)
        outputs = []
        out = self.flush(enable_amp=enable_amp, edge_dilation=edge_dilation, depth_aa=depth_aa_bool)
        if out is not None:
            for i in range(out.shape[0]):
                normalized_depth = self.minmax_normalize_chw(out[i])
                if normalized_depth is not None:
                    outputs.append(normalized_depth)
        outputs += self.flush_minmax_normalize()
        return outputs

    def _flush(self, enable_amp=True):
        results = []
        while self.output_frame_count < self.input_frame_count:
            ret = self.model.infer(None, use_amp=enable_amp)
            if ret is None:
                continue
            results += ret
            self.output_frame_count += len(ret)
        if results:
            unpad = self.output_frame_count - self.input_frame_count
            if unpad > 0:
                assert unpad <= len(results)
                results = results[:-unpad]
            return results
        else:
            return []

    order: int = 50

    @classmethod
    def get_name(cls):
        return "VideoDepthAnything"

    @classmethod
    def is_desktop_supported(cls) -> bool:
        return False

    @classmethod
    def has_auto_download(cls, model_type: str) -> bool:
        return model_type in {"VDA_S", "VDA_Metric_S"}

    def is_image_supported(self):
        return False

    def is_streaming(self):
        return True

    @classmethod
    def has_checkpoint_file(cls, model_type):
        return cls.supported(model_type) and path.exists(MODEL_FILES[model_type])

    @classmethod
    def supported(cls, model_type):
        return model_type in MODEL_FILES

    @classmethod
    def get_model_types(cls) -> list[str]:
        return list(MODEL_FILES.keys())

    @classmethod
    def get_model_path(cls, model_type):
        return MODEL_FILES[model_type]

    def is_metric(self):
        if not self.metric_depth:
            return False
        return not self.force_disparity

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return False  # TODO

    @classmethod
    def is_depth_aa_supported(cls, model_type: str) -> bool:
        return model_type in AA_SUPPORT_MODELS

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/Video-Depth-Anything_iw3:main", "VideoDepthAnythingOnline")

    def compile(self):
        if self.model_backup is None:
            self.model_backup = (self.model.head, self.model.pretrained.get_intermediate_layers)
            self.model.head = compile_model(self.model.head)
            self.model.pretrained.get_intermediate_layers = compile_function(
                self.model.pretrained.get_intermediate_layers, device=self.device
            )

    def clear_compiled_model(self):
        if self.model_backup is not None:
            self.model.head, self.model.pretrained.get_intermediate_layers = self.model_backup
            self.model_backup = None


def _test():
    import random

    import torchvision.transforms.functional as TF
    from PIL import Image

    N = 111

    model = VideoDepthAnythingModel("VDA_S")
    model.enable_ema(decay=0.99, buffer_size=10)
    model.load(gpu=0)
    # model.compile()
    im = Image.open("tests/images/dog_448.png").convert("RGB")
    x = TF.to_tensor(im).unsqueeze(0).to(model.device)
    reset_pts = set([i + N // 2 for i in range(N // 2) if random.uniform(0, 1) < 0.1])
    outputs = []
    for i in range(N):
        depth = model.infer(x, streaming=True)
        outputs += model.minmax_normalize(depth)
        if i in reset_pts:
            flush_depth = model.flush()
            outputs += model.minmax_normalize(flush_depth)
            outputs += model.flush_minmax_normalize()
            model.reset()
    flush_depth = model.flush()
    outputs += model.minmax_normalize(flush_depth)
    outputs += model.flush_minmax_normalize()

    assert len(outputs) == N
    print(len(outputs), outputs[0].shape)


if __name__ == "__main__":
    _test()
