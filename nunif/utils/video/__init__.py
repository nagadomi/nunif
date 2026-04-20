from .color_transform import TensorFrame
from .frame_callback_pool import (
    FrameCallbackPool,
    get_source_dtype,
    to_tensor,
)
from .hwaccel import (
    HW_DEVICES,
    create_hwaccel,
)
from .initializer import initialize_library, pyav_init_cuda_primary_context
from .metadata import VideoMetadata
from .offload_frame import OffloadFrame
from .output_config import VideoOutputConfig
from .processor import (
    export_audio,
    generate_video,
    hook_frame,
    make_error_file_path,
    process_video,
    process_video_keyframes,
    sample_frames,
)
from .utils import (
    LIBH264,
    RGB_8BIT,
    RGB_16BIT,
    VIDEO_EXTENSIONS,
    get_default_video_codec,
    has_nvenc,
    has_qsv,
    list_videos,
    pix_fmt_requires_16bit,
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
    "OffloadFrame",
    "VideoOutputConfig",
    "export_audio",
    "generate_video",
    "hook_frame",
    "make_error_file_path",
    "process_video",
    "process_video_keyframes",
    "sample_frames",
    "LIBH264",
    "RGB_8BIT",
    "RGB_16BIT",
    "VIDEO_EXTENSIONS",
    "get_default_video_codec",
    "has_nvenc",
    "has_qsv",
    "list_videos",
    "pix_fmt_requires_16bit",
]

# Initialize library defaults (MIME types, logging, env vars)
initialize_library()
