import os
import threading
from fractions import Fraction
from typing import List

import av
import torch
from av.codec.hwaccel import hwdevices_available

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


DEFAULT_VAAPI_DEVICE = os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128")

_VIDEO_CODEC_CACHE_LOCK = threading.Lock()
_VIDEO_CODEC_CACHE: dict[tuple[str, int | None], bool] = {}


def clear_video_codec_cache() -> None:
    """Clear the cached results of video encoder probe checks."""
    with _VIDEO_CODEC_CACHE_LOCK:
        _VIDEO_CODEC_CACHE.clear()


def is_vaapi_codec(codec) -> bool:
    return isinstance(codec, str) and codec.endswith("_vaapi")


def _probe_video_codec(codec: str, gpu_id: int | None = None) -> bool:
    """Verify that a video encoder can open and encode a 256x256 frame."""
    if codec not in av.codec.codecs_available:
        return False

    from typing import cast

    from av.video.codeccontext import VideoCodecContext

    ctx: VideoCodecContext | None = None
    try:
        options: dict[str, str] = {}
        if codec.endswith(("_nvenc", "_qsv", "_amf")):
            ctx_pix_fmt = "nv12"
            frame_pix_fmt = "nv12"
            if codec.endswith("_nvenc") and gpu_id is not None:
                options["gpu"] = str(gpu_id)
        else:
            ctx_pix_fmt = "yuv420p"
            frame_pix_fmt = "yuv420p"
            if codec == "libx265":
                options["x265-params"] = "log-level=error"

        ctx = cast(VideoCodecContext, av.codec.CodecContext.create(codec, "w"))
        ctx.width = 256
        ctx.height = 256
        ctx.time_base = Fraction(1, 30)
        ctx.framerate = Fraction(30, 1)
        ctx.pix_fmt = ctx_pix_fmt
        if options:
            ctx.options = options
        ctx.open(strict=False)

        frame = av.VideoFrame(256, 256, frame_pix_fmt)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))

        packets = list(ctx.encode(frame))
        packets.extend(ctx.encode(None))
        return len(packets) > 0
    except Exception:
        return False
    finally:
        del ctx


def _has_video_codec_cached(codec: str, cache_gpu_id: int | None) -> bool:
    key = (codec, cache_gpu_id)
    with _VIDEO_CODEC_CACHE_LOCK:
        cached = _VIDEO_CODEC_CACHE.get(key)
        if cached is not None:
            return cached
        ok = _probe_video_codec(codec, gpu_id=cache_gpu_id)
        _VIDEO_CODEC_CACHE[key] = ok
        return ok


def has_video_codec(codec: str, gpu_id: int | None = None) -> bool:
    """Return whether the given video encoder is usable on this machine (cached)."""
    if not codec or codec not in av.codec.codecs_available:
        return False

    # VAAPI is only enabled in custom FFmpeg/PyAV builds and depends on render node paths,
    # so only check codec and hwdevice availability without running a dummy encode probe.
    if is_vaapi_codec(codec):
        return "vaapi" in hwdevices_available()

    if codec.endswith("_nvenc"):
        if gpu_id is not None and gpu_id >= 0:
            return _has_video_codec_cached(codec, gpu_id)
        with _VIDEO_CODEC_CACHE_LOCK:
            cached = _VIDEO_CODEC_CACHE.get((codec, None))
            if cached is not None:
                return cached
        ok = False
        for dev_id in range(4):
            if _has_video_codec_cached(codec, dev_id):
                ok = True
                break
        with _VIDEO_CODEC_CACHE_LOCK:
            _VIDEO_CODEC_CACHE[(codec, None)] = ok
        return ok

    if codec.endswith(("_qsv", "_amf")):
        return _has_video_codec_cached(codec, None)

    return True


def has_nvenc(gpu_id: int | None = None) -> bool:
    return any(has_video_codec(c, gpu_id=gpu_id) for c in ("h264_nvenc", "hevc_nvenc"))


def has_qsv() -> bool:
    return any(has_video_codec(c) for c in ("h264_qsv", "hevc_qsv"))


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
    return any(has_video_codec(c) for c in ("h264_amf", "hevc_amf", "av1_amf"))


def has_vaapi() -> bool:
    return any(has_video_codec(c) for c in ("h264_vaapi", "hevc_vaapi", "av1_vaapi"))


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
