import gc
import struct
from fractions import Fraction
from typing import Any, cast

import av
import torch
import torch.nn.functional as F
from av.video.codeccontext import VideoCodecContext
from av.video.frame import CudaContext, PictureType
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc
from packaging.version import Version

from nunif.utils.video import utils as VU
from nunif.utils.video.color_transform import OutputTransform
from nunif.utils.video.hwaccel import HW_DEVICES, create_hwaccel

TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)

STREAM_VIDEO_CODECS = (
    "libx264",
    "libopenh264",
    "libx265",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "h264_amf",
    "hevc_amf",
    "h264_vaapi",
    "hevc_vaapi",
)
NVENC_CODECS = ("h264_nvenc", "hevc_nvenc")
VAAPI_CODECS = ("h264_vaapi", "hevc_vaapi")
AMF_CODECS = ("h264_amf", "hevc_amf")
QSV_CODECS = ("h264_qsv", "hevc_qsv")
JPEG_VIDEO_CODECS = ("jpeg", "mjpeg")
JPEG_GPU_VIDEO_CODECS = ("jpeg-gpu", "jpeg_gpu")


def is_gpu_jpeg_available() -> bool:
    return ENABLE_GPU_JPEG and torch.cuda.is_available() and not VU.IS_ROCM


def get_default_stream_video_codec() -> str:
    if "libx264" in av.codec.codecs_available:
        return "libx264"
    if "libopenh264" in av.codec.codecs_available:
        return "libopenh264"
    return "jpeg"


DEFAULT_STREAM_VIDEO_CODEC = get_default_stream_video_codec()


def normalize_video_codec(video_codec: str | None) -> str:
    if not video_codec:
        return get_default_stream_video_codec()
    value = str(video_codec).strip()
    lower = value.lower()
    if lower in JPEG_VIDEO_CODECS:
        return "jpeg"
    if lower in JPEG_GPU_VIDEO_CODECS:
        return "jpeg-gpu"
    return value


def is_jpeg_codec(video_codec: str | None) -> bool:
    return normalize_video_codec(video_codec) in {"jpeg", "jpeg-gpu"}


def get_codec_family(video_codec: str) -> str | None:
    if video_codec in {"h264_nvenc", "h264_vaapi", "h264_amf", "h264_qsv", "libx264", "libx264rgb", "libopenh264"}:
        return "h264"
    if video_codec in {"hevc_nvenc", "hevc_vaapi", "hevc_amf", "hevc_qsv", "libx265"}:
        return "hevc"
    return None


def get_available_stream_video_codecs(gpu_id: int | None = None) -> list[str]:
    has_vaapi = "vaapi" in HW_DEVICES
    codecs: list[str] = []
    for codec in STREAM_VIDEO_CODECS:
        if codec not in av.codec.codecs_available:
            continue
        if codec in VAAPI_CODECS and not has_vaapi:
            continue
        if not VU.has_video_codec(codec, gpu_id=gpu_id):
            continue
        codecs.append(codec)
    codecs.append("jpeg")
    if is_gpu_jpeg_available():
        codecs.append("jpeg-gpu")
    return codecs


def validate_stream_video_codec(video_codec: str | None, gpu_id: int | None = None) -> str:
    vc = normalize_video_codec(video_codec)
    if vc == "jpeg":
        return vc
    if vc == "jpeg-gpu":
        if not ENABLE_GPU_JPEG:
            raise ValueError("video_codec='jpeg-gpu' requires PyTorch/torchvision >= 2.7")
        if not (torch.cuda.is_available() and not VU.IS_ROCM):
            raise ValueError("video_codec='jpeg-gpu' requires an NVIDIA CUDA GPU")
        return vc
    if get_codec_family(vc) is None:
        supported = ", ".join(get_available_stream_video_codecs(gpu_id=gpu_id))
        raise ValueError(f"Unsupported stream video codec: {vc}. Supported codecs: {supported}")
    if vc in VAAPI_CODECS and "vaapi" not in HW_DEVICES:
        raise ValueError(f"VAAPI hardware device is not available for '{vc}'")
    if vc not in av.codec.codecs_available or not VU.has_video_codec(vc, gpu_id=gpu_id):
        raise ValueError(f"Video encoder '{vc}' is not available on this system")
    return vc


def get_stream_codec_families(video_codec: str | None) -> list[str]:
    vc = resolve_stream_video_codec(video_codec=video_codec)
    if vc is None:
        return []
    family = get_codec_family(vc)
    return [family] if family is not None else []


def resolve_stream_video_codec(
    video_codec: str | None = None,
    client_codecs: list[str] | None = None,
) -> str | None:
    vc = normalize_video_codec(video_codec)
    if is_jpeg_codec(vc):
        return None
    family = get_codec_family(vc)
    if family is None or vc not in av.codec.codecs_available:
        return None
    if vc in VAAPI_CODECS and "vaapi" not in HW_DEVICES:
        return None
    if client_codecs is not None:
        client_set = {c.lower() for c in client_codecs if c}
        if family not in client_set:
            return None
    return vc


def resolve_stream_codec_config(args: Any) -> tuple[str, list[str], str | None]:
    if args is None:
        video_codec = get_default_stream_video_codec()
        vaapi_device = None
    else:
        video_codec = normalize_video_codec(getattr(args, "video_codec", None))
        vaapi_device = getattr(args, "vaapi_device", None)
    available = get_stream_codec_families(video_codec)
    return video_codec, available, vaapi_device


def map_stream_quality_to_qp(quality: int, video_codec: str) -> int:
    q = max(1, min(100, int(quality)))
    if video_codec in {"h264_vaapi", "hevc_vaapi", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf"}:
        return round(14 + (100 - q) * 24 / 99)
    if video_codec in {"hevc_nvenc", "libx265"}:
        return round(14 + (100 - q) * 24 / 99)
    return round(12 + (100 - q) * 24 / 99)


def map_stream_quality_to_bitrate(
    quality: int,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> int:
    q = max(1, min(100, int(quality)))
    base_bitrate = 2_000_000 + (q - 1) * (26_000_000 / 99)
    pixel_rate_scale = max(0.5, min(4.0, (width * height * max(1, fps)) / (1920 * 1080 * 30)))
    return round(base_bitrate * pixel_rate_scale)


def _iter_annexb_nals(data: bytes) -> list[bytes]:
    nals: list[bytes] = []
    n = len(data)
    i = 0
    while i < n:
        if i + 3 <= n and data[i : i + 3] == b"\x00\x00\x01":
            start = i + 3
        elif i + 4 <= n and data[i : i + 4] == b"\x00\x00\x00\x01":
            start = i + 4
        else:
            i += 1
            continue

        j = data.find(b"\x00\x00\x01", start)
        if j == -1:
            nals.append(data[start:])
            break
        end = j
        if end > start and data[end - 1] == 0:
            end -= 1
        nals.append(data[start:end])
        i = j
    return nals


def _unescape_rbsp(nal: bytes) -> bytes:
    out = bytearray()
    zero_count = 0
    for b in nal:
        if zero_count >= 2 and b == 0x03:
            zero_count = 0
            continue
        out.append(b)
        if b == 0:
            zero_count += 1
        else:
            zero_count = 0
    return bytes(out)


def _reverse_bits_u32(val: int) -> int:
    res = 0
    for _ in range(32):
        res = (res << 1) | (val & 1)
        val >>= 1
    return res


def extract_webcodecs_codec_string(annexb_bytes: bytes, codec_family: str) -> str | None:
    for nal in _iter_annexb_nals(annexb_bytes):
        if not nal:
            continue
        if codec_family == "h264":
            nal_type = nal[0] & 0x1F
            if nal_type == 7 and len(nal) >= 4:
                return f"avc1.{nal[1]:02x}{nal[2]:02x}{nal[3]:02x}"
        elif codec_family == "hevc":
            nal_type = (nal[0] >> 1) & 0x3F
            if nal_type == 33:
                rbsp = _unescape_rbsp(nal)
                if len(rbsp) < 15:
                    continue
                # Skip 2-byte NAL header and 1-byte VPS id / sub-layer / nesting byte.
                ptl = rbsp[3:15]
                space = (ptl[0] >> 6) & 0x03
                tier = "H" if (ptl[0] & 0x20) else "L"
                profile_idc = ptl[0] & 0x1F
                space_prefix = ("", "A", "B", "C")[space]
                compat_raw = struct.unpack(">I", ptl[1:5])[0]
                compat_rev = _reverse_bits_u32(compat_raw)
                level_idc = ptl[11]
                constraints = list(ptl[5:11])
                while constraints and constraints[-1] == 0:
                    constraints.pop()
                parts = [
                    "hev1",
                    f"{space_prefix}{profile_idc}",
                    f"{compat_rev:x}",
                    f"{tier}{level_idc}",
                ]
                parts.extend(f"{b:02X}" for b in constraints)
                return ".".join(parts)
    return None


def _build_codec_options(video_codec: str, qp: int) -> dict[str, str]:
    qp_str = str(qp)
    if video_codec in NVENC_CODECS:
        return {
            "preset": "p1",
            "tune": "ull",
            "rc": "vbr",
            "cq": qp_str,
            "zerolatency": "1",
            "delay": "0",
            "forced-idr": "1",
            "rc-lookahead": "0",
        }
    if video_codec in VAAPI_CODECS:
        return {
            "rc_mode": "CQP",
            "qp": qp_str,
        }
    if video_codec in AMF_CODECS:
        return {
            "usage": "ultralowlatency",
            "rc": "cqp",
            "qp_i": qp_str,
            "qp_p": qp_str,
        }
    if video_codec in QSV_CODECS:
        return {
            "preset": "veryfast",
            "look_ahead": "0",
            "global_quality": qp_str,
        }
    if video_codec == "libopenh264":
        return {
            "allow_skip_frames": "0",
            "rc_mode": "bitrate",
        }
    if video_codec == "libx265":
        return {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": qp_str,
            "forced-idr": "1",
            "x265-params": "log-level=error",
        }
    return {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": qp_str,
        "forced-idr": "1",
    }


def _pad_to_even(tensor: torch.Tensor) -> torch.Tensor:
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_w = width % 2
    pad_h = height % 2
    if pad_w == 0 and pad_h == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


class VideoStreamEncoder:
    def __init__(
        self,
        video_codec: str,
        fps: int = 30,
        quality: int = 90,
        vaapi_device: str | None = None,
    ) -> None:
        family = get_codec_family(video_codec)
        if family is None:
            raise ValueError(f"Unsupported stream video codec: {video_codec}")

        self.video_codec = video_codec
        self.codec_family = family
        self.fps = max(1, int(fps))
        self.quality = max(1, min(100, int(quality)))
        self.vaapi_device = vaapi_device

        self._ctx: VideoCodecContext | None = None
        self._output_transform: OutputTransform | None = None
        self._cuda_context: CudaContext | None = None
        self._width = 0
        self._height = 0
        self._use_cuda_dlpack = False
        self._codec_string: str | None = None
        self._frame_count = 0
        self._pts_base_us: int | None = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec_string(self) -> str:
        if self._codec_string:
            return self._codec_string
        if self.codec_family == "hevc":
            return "hev1.1.6.L120.90"
        return "avc1.42001f"

    def _setup(self, width: int, height: int, device: torch.device) -> None:
        self.close()

        use_nvenc = self.video_codec in NVENC_CODECS
        use_vaapi = self.video_codec in VAAPI_CODECS
        use_yuv420p = self.video_codec in {"libx265", "libopenh264"}
        use_cuda_dlpack = use_nvenc and VU.is_nvidia_gpu(device)

        hwaccel = None
        if use_vaapi:
            if "vaapi" not in HW_DEVICES:
                raise ValueError(f"VAAPI hardware acceleration is not available for '{self.video_codec}'")
            hwaccel = create_hwaccel(
                "vaapi",
                device_id=None,
                disable_software_fallback=True,
                vaapi_device=self.vaapi_device,
            )

        ctx = cast(VideoCodecContext, av.CodecContext.create(self.video_codec, "w", hwaccel=hwaccel))
        ctx.width = width
        ctx.height = height
        ctx.time_base = Fraction(1, 1_000_000)
        ctx.framerate = Fraction(self.fps, 1)
        ctx.max_b_frames = 0
        ctx.gop_size = max(1, self.fps * 2)
        ctx.bit_rate = (
            map_stream_quality_to_bitrate(self.quality, width=width, height=height, fps=self.fps)
            if self.video_codec == "libopenh264"
            else 0
        )
        ctx.colorspace = Colorspace.ITU709
        ctx.color_primaries = ColorPrimaries.BT709
        ctx.color_trc = ColorTrc.BT709
        ctx.color_range = ColorRange.MPEG

        if use_cuda_dlpack:
            ctx.pix_fmt = "cuda"
        elif use_vaapi:
            ctx.pix_fmt = "vaapi"
        elif use_yuv420p:
            ctx.pix_fmt = "yuv420p"
        else:
            ctx.pix_fmt = "nv12"

        qp = map_stream_quality_to_qp(self.quality, self.video_codec)
        ctx.options = _build_codec_options(self.video_codec, qp)

        if use_vaapi:
            ctx.open(strict=False)

        cuda_ctx: CudaContext | None = None
        if use_cuda_dlpack:
            device_id = device.index if device.index is not None else 0
            cuda_ctx = CudaContext(device_id=device_id, current_ctx=True, primary_ctx=False)

        transform_pix_fmt = "yuv420p" if use_yuv420p else "nv12"
        output_transform = OutputTransform(
            dst_pix_fmt=transform_pix_fmt,
            dst_colorspace=Colorspace.ITU709,
            dst_color_primaries=ColorPrimaries.BT709,
            dst_color_trc=ColorTrc.BT709,
            dst_color_range=ColorRange.MPEG,
            cuda_context=cuda_ctx,
        )

        self._ctx = ctx
        self._output_transform = output_transform
        self._cuda_context = cuda_ctx
        self._width = width
        self._height = height
        self._use_cuda_dlpack = use_cuda_dlpack
        self._codec_string = None
        self._frame_count = 0
        self._pts_base_us = None

    def encode(
        self,
        tensor: torch.Tensor,
        timestamp_us: int,
        force_keyframe: bool = False,
    ) -> tuple[bytes, bool, str, int, int] | None:
        tensor = _pad_to_even(tensor)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        use_cuda_dlpack = (self.video_codec in NVENC_CODECS) and VU.is_nvidia_gpu(tensor.device)

        if (
            self._ctx is None
            or self._output_transform is None
            or self._width != width
            or self._height != height
            or self._use_cuda_dlpack != use_cuda_dlpack
        ):
            self._setup(width, height, tensor.device)
            force_keyframe = True

        assert self._ctx is not None
        assert self._output_transform is not None

        if tensor.dtype not in (torch.float32, torch.float16):
            tensor = tensor.float()

        vframe = self._output_transform.transform(tensor)
        if self._pts_base_us is None:
            self._pts_base_us = int(timestamp_us)
        vframe.pts = max(0, int(timestamp_us) - self._pts_base_us)
        if force_keyframe:
            vframe.pict_type = PictureType.I

        packets = self._ctx.encode(vframe)
        self._frame_count += 1
        if self._use_cuda_dlpack and (self._frame_count % 60 == 0):
            gc.collect(0)

        if not packets:
            return None

        payload = b"".join(bytes(pkt) for pkt in packets)
        if not payload:
            return None

        is_keyframe = any(pkt.is_keyframe for pkt in packets)
        if is_keyframe or self._codec_string is None:
            detected = extract_webcodecs_codec_string(payload, self.codec_family)
            if detected:
                self._codec_string = detected

        return payload, is_keyframe, self.codec_string, self._width, self._height

    def close(self) -> None:
        if self._ctx is not None:
            try:
                if self._ctx.is_open:
                    self._ctx.encode(None)
            except Exception:
                pass
        self._ctx = None
        self._output_transform = None
        self._cuda_context = None
        self._width = 0
        self._height = 0
        self._use_cuda_dlpack = False
        self._codec_string = None
        self._frame_count = 0
        self._pts_base_us = None


__all__ = [
    "DEFAULT_STREAM_VIDEO_CODEC",
    "ENABLE_GPU_JPEG",
    "STREAM_VIDEO_CODECS",
    "VideoStreamEncoder",
    "extract_webcodecs_codec_string",
    "get_available_stream_video_codecs",
    "get_codec_family",
    "get_default_stream_video_codec",
    "get_stream_codec_families",
    "is_gpu_jpeg_available",
    "is_jpeg_codec",
    "map_stream_quality_to_bitrate",
    "map_stream_quality_to_qp",
    "normalize_video_codec",
    "resolve_stream_codec_config",
    "resolve_stream_video_codec",
    "validate_stream_video_codec",
]
