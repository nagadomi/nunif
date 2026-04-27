import os
from typing import List

import av
import torch

IS_ROCM = getattr(torch.version, "hip", None) is not None


def is_nvidia_gpu(device: torch.device | None) -> bool:
    if device is None:
        return False
    return device.type == "cuda" and not IS_ROCM


def is_discrete_device(device: torch.device | None) -> bool:
    if device is None:
        return False  # assume cpu

    # mps assumes unified memory model
    return device.type in {"cuda", "xpu"}


VIDEO_EXTENSIONS = [
    ".mp4",
    ".m4v",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".avi",
    ".wmv",
    ".mov",
    ".flv",
    ".webm",
    ".asf",
    ".vob",
    ".divx",
    ".3gp",
    ".ogv",
    ".3g2",
    ".m2ts",
    ".ts",
    ".rm",
]

RGB_8BIT = "rgb24"
RGB_16BIT = "gbrp16le"


def _get_libh264() -> str:
    if "libx264" in av.codecs_available:
        return "libx264"
    elif "libopenh264" in av.codecs_available:
        return "libopenh264"
    return ""


LIBH264 = _get_libh264()


def has_nvenc() -> bool:
    return "h264_nvenc" in av.codec.codecs_available and "hevc_nvenc" in av.codec.codecs_available


def has_qsv() -> bool:
    return "h264_qsv" in av.codec.codecs_available and "hevc_qsv" in av.codec.codecs_available


def list_videos(directory: str, extensions: List[str] = VIDEO_EXTENSIONS) -> List[str]:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[-1].lower() in extensions
    )


def get_default_video_codec(container_format: str) -> str:
    if container_format in {"mp4", "mkv"}:
        return LIBH264
    elif container_format == "avi":
        return "utvideo"
    else:
        raise ValueError(f"Unsupported container format: {container_format}")


def pix_fmt_requires_16bit(pix_fmt: str) -> bool:
    return pix_fmt in {
        "yuv420p10le",
        "p010le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "yuv444p16le",
        RGB_16BIT,
        "gbrp12le",
        "gbrp10le",
        "rgb48le",
    }
