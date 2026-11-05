import os
from fractions import Fraction
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


def has_vp9() -> bool:
    return "libvpx-vp9" in av.codec.codecs_available


def is_vp9_alpha(stream: av.video.stream.VideoStream) -> bool:
    if stream.codec_context.name != "vp9":
        return False
    alpha_mode = stream.metadata.get("alpha_mode")
    if not alpha_mode and stream.container is not None:
        alpha_mode = stream.container.metadata.get("alpha_mode")
    return alpha_mode in {"1", "true", "True"}


def has_amf() -> bool:
    return "h264_amf" in av.codec.codecs_available and "hevc_amf" in av.codec.codecs_available


def has_vaapi() -> bool:
    return "h264_vaapi" in av.codec.codecs_available and "hevc_vaapi" in av.codec.codecs_available


def list_videos(directory: str, extensions: List[str] = VIDEO_EXTENSIONS) -> List[str]:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[-1].lower() in extensions
    )


def get_default_video_codec(container_format: str) -> str:
    if container_format in {"mp4", "mkv"}:
        return LIBH264
    elif container_format == "avi":
        return "utvideo"
    elif container_format == "webm":
        return "libvpx-vp9"
    else:
        raise ValueError(f"Unsupported container format: {container_format}")


ALPHA_PIX_FORMATS = {
    "yuva420p",
    "yuva422p",
    "yuva444p",
    "yuva420p9le",
    "yuva422p9le",
    "yuva444p9le",
    "yuva420p10le",
    "yuva422p10le",
    "yuva444p10le",
    "yuva422p12le",
    "yuva444p12le",
    "yuva420p16le",
    "yuva422p16le",
    "yuva444p16le",
    "ayuv",
    "ayuv64le",
    "rgba",
    "bgra",
    "argb",
    "abgr",
    "rgba64le",
    "bgra64le",
    "gbrap",
    "gbrap10le",
    "gbrap12le",
    "gbrap14le",
    "gbrap16le",
    "ya8",
    "ya16le",
}


def pix_fmt_has_alpha(pix_fmt: str | av.VideoFormat) -> bool:
    if isinstance(pix_fmt, av.VideoFormat):
        return pix_fmt.name.lower() in ALPHA_PIX_FORMATS or any(c.is_alpha for c in pix_fmt.components)
    return pix_fmt.lower() in ALPHA_PIX_FORMATS


def pix_fmt_requires_16bit(pix_fmt: str) -> bool:
    return pix_fmt.lower() in {
        "yuv420p10le",
        "p010le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "yuv420p16le",
        "yuv422p16le",
        "yuv444p16le",
        RGB_16BIT,
        "gbrp12le",
        "gbrp10le",
        "rgb48le",
        "yuva420p9le",
        "yuva422p9le",
        "yuva444p9le",
        "yuva420p10le",
        "yuva422p10le",
        "yuva444p10le",
        "yuva422p12le",
        "yuva444p12le",
        "yuva420p16le",
        "yuva422p16le",
        "yuva444p16le",
        "ayuv64le",
        "rgba64le",
        "bgra64le",
        "gbrap10le",
        "gbrap12le",
        "gbrap14le",
        "gbrap16le",
        "ya16le",
    }


def av_rational_to_fraction(value: av.AVRational | None):
    if value is None:
        return None
    return Fraction(value.num, value.den)


def vpx_preset_to_cpu_used(preset: str | None) -> str:
    if preset in {"fast", "faster", "veryfast", "superfast", "ultrafast"}:
        return "4"
    elif preset in {"slow", "slower", "veryslow", "placebo"}:
        return "1"
    else:
        return "2"
