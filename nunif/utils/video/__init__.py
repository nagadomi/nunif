from .color_transform import (
    TensorFrame,
    get_source_dtype,
    to_ndarray,
    to_tensor,
)
from .decoder import VideoDecoder
from .frame_callback_pool import (
    FrameCallbackPool,
)
from .frame_offload import OffloadedFrame, OffloadResourceManager
from .hwaccel import (
    HW_DEVICES,
    create_hwaccel,
)
from .initializer import initialize_library, pyav_init_cuda_primary_context
from .metadata import VideoMetadata
from .output_config import VideoOutputConfig
from .processor import (
    export_audio,
    generate_video,
    hook_frame,
    make_error_file_path,
    process_video,
    sample_frames,
)
from .utils import (
    ALPHA_PIX_FORMATS,
    LIBH264,
    RGB_8BIT,
    RGB_16BIT,
    VIDEO_EXTENSIONS,
    get_default_video_codec,
    has_amf,
    has_nvenc,
    has_qsv,
    has_vaapi,
    has_vp9,
    list_videos,
    pix_fmt_has_alpha,
    pix_fmt_requires_16bit,
    vpx_preset_to_cpu_used,
)

__all__ = [
    "TensorFrame",
    "FrameCallbackPool",
    "get_source_dtype",
    "to_frame",
    "to_ndarray",
    "to_tensor",
    "HW_DEVICES",
    "create_hwaccel",
    "initialize_library",
    "pyav_init_cuda_primary_context",
    "VideoMetadata",
    "OffloadedFrame",
    "OffloadResourceManager",
    "VideoOutputConfig",
    "export_audio",
    "generate_video",
    "hook_frame",
    "make_error_file_path",
    "process_video",
    "sample_frames",
    "VideoDecoder",
    "ALPHA_PIX_FORMATS",
    "LIBH264",
    "RGB_8BIT",
    "RGB_16BIT",
    "VIDEO_EXTENSIONS",
    "get_default_video_codec",
    "has_amf",
    "has_nvenc",
    "has_qsv",
    "has_vaapi",
    "has_vp9",
    "list_videos",
    "pix_fmt_has_alpha",
    "pix_fmt_requires_16bit",
    "vpx_preset_to_cpu_used",
]

# Initialize library defaults (MIME types, logging, env vars)
initialize_library()
