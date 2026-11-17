import av
import pytest
import torch

from nunif.utils.video import HW_DEVICES, create_hwaccel, has_video_codec
from nunif.utils.video.utils import is_nvidia_gpu


def is_cuda_hwaccel_available() -> bool:
    """Return whether CUDA hardware decode/frame transfer is available on an NVIDIA GPU."""
    if "cuda" not in HW_DEVICES or not torch.cuda.is_available():
        return False
    return is_nvidia_gpu(torch.device("cuda:0"))


def is_nvenc_available(codec: str = "h264_nvenc", require_cuda_hwaccel: bool = True) -> bool:
    """Return whether the given NVENC encoder (and optionally CUDA hwaccel) is available."""
    if require_cuda_hwaccel and not is_cuda_hwaccel_available():
        return False
    return torch.cuda.is_available() and has_video_codec(codec)


def is_vaapi_available(codec: str = "h264_vaapi") -> bool:
    """Return whether VAAPI hardware context initialization and the given codec are available."""
    if "vaapi" not in HW_DEVICES or not has_video_codec(codec):
        return False
    try:
        hwaccel = create_hwaccel(device="vaapi", disable_software_fallback=True)
        av.codec.CodecContext.create(codec, "w", hwaccel=hwaccel)
        return True
    except Exception:
        return False


def nvenc_required(codec: str = "h264_nvenc", require_cuda_hwaccel: bool = True) -> pytest.MarkDecorator:
    """Skip test unless the specified NVENC encoder is available."""
    return pytest.mark.skipif(
        not is_nvenc_available(codec, require_cuda_hwaccel=require_cuda_hwaccel),
        reason=f"{codec} not available on this host",
    )


def vaapi_required(codec: str = "h264_vaapi") -> pytest.MarkDecorator:
    """Skip test unless VAAPI hwaccel and the specified VAAPI encoder are available."""
    return pytest.mark.skipif(
        not is_vaapi_available(codec),
        reason=f"{codec} / VAAPI hwaccel not available on this host",
    )


cuda_hwaccel_required = pytest.mark.skipif(
    not is_cuda_hwaccel_available(),
    reason="CUDA hwaccel not available on this host",
)

nvenc_h264_required = nvenc_required("h264_nvenc")
nvenc_h265_required = nvenc_required("hevc_nvenc")
vaapi_h264_required = vaapi_required("h264_vaapi")
vaapi_h265_required = vaapi_required("hevc_vaapi")
