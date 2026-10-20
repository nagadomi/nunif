import io
import os
import sys
import threading
from typing import Any, Optional, Sequence, Union

import av
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
from PIL import Image as PILImage

from .dir_config import PUBLIC_DIR
from .stereo_detector import (
    FLAT,
    SBS_180,
    SBS_FULL,
    SBS_FULL_CROSS,
    SBS_HALF,
    TB_FULL,
    TB_HALF,
    detect_stereo_format,
)

THUMBNAIL_SIZE = 64
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}


VIDEO_THUMBNAIL_PERCENT = 0.20
VIDEO_THUMBNAIL_MIN_SEC = 5.0
VIDEO_THUMBNAIL_MAX_SEC = 60.0


def crop_left_eye(img: PILImage.Image, fmt: str) -> PILImage.Image:
    """Crops the left eye view from a stereo image based on the stereo format."""
    width, height = img.size
    if fmt in [SBS_FULL, SBS_HALF, SBS_180]:
        left_view = img.crop((0, 0, width // 2, height))
        if fmt == SBS_HALF:
            left_view = left_view.resize((left_view.width * 2, left_view.height), PILImage.Resampling.LANCZOS)
        return left_view
    elif fmt in [SBS_FULL_CROSS]:
        left_view = img.crop((width // 2, 0, width, height))
        return left_view
    elif fmt in [TB_FULL, TB_HALF]:
        left_view = img.crop((0, 0, width, height // 2))
        if fmt == TB_HALF:
            left_view = left_view.resize((left_view.width, left_view.height * 2), PILImage.Resampling.LANCZOS)
        return left_view
    else:
        return img


def render_thumbnail(img: PILImage.Image, stereo_fmt: str) -> bytes:
    """Crops, resizes, and encodes a source image into a 64x64 WebP thumbnail."""
    img = img.convert("RGBA")
    left_view = crop_left_eye(img, stereo_fmt)
    left_view.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), PILImage.Resampling.LANCZOS)
    canvas = PILImage.new("RGBA", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), (0, 0, 0, 0))
    offset = ((THUMBNAIL_SIZE - left_view.width) // 2, (THUMBNAIL_SIZE - left_view.height) // 2)
    canvas.paste(left_view, offset)
    output = io.BytesIO()
    canvas.save(output, format="WEBP", quality=80)
    return output.getvalue()


def compute_video_thumbnail_target_sec(duration_sec: Optional[float]) -> float:
    """Computes target seek time in seconds for video thumbnail generation.

    Uses 20% of duration clamped to [5.0s, 60.0s]. If duration is shorter than 5.0s,
    it scales down proportionally to 20% of the duration.
    """
    if duration_sec is not None and duration_sec > 0:
        if duration_sec <= VIDEO_THUMBNAIL_MIN_SEC:
            return duration_sec * VIDEO_THUMBNAIL_PERCENT
        return max(VIDEO_THUMBNAIL_MIN_SEC, min(duration_sec * VIDEO_THUMBNAIL_PERCENT, VIDEO_THUMBNAIL_MAX_SEC))
    return VIDEO_THUMBNAIL_MIN_SEC


def _resolve_and_render_thumbnail(
    img: PILImage.Image,
    stereo_fmt: Optional[str] = None,
    path_or_candidates: Union[str, Sequence[str], None] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> bytes:
    fmt = stereo_fmt or FLAT
    if fmt == FLAT and (path_or_candidates or cache_db is not None):
        fmt = (
            detect_stereo_format(
                path_or_candidates,
                image=img,
                cache_db=cache_db,
                cache_key=cache_key,
                fallback_flat=True,
            )
            or FLAT
        )
    return render_thumbnail(img, fmt)


def create_thumbnail_data(
    fp: io.BytesIO,
    stereo_fmt: Optional[str] = None,
    path_or_candidates: Union[str, Sequence[str], None] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> Optional[bytes]:
    """Generates a 64x64 WebP thumbnail for an image stream."""
    try:
        with PILImage.open(fp) as img:
            return _resolve_and_render_thumbnail(
                img,
                stereo_fmt=stereo_fmt,
                path_or_candidates=path_or_candidates,
                cache_db=cache_db,
                cache_key=cache_key,
            )
    except Exception as e:
        print(f"Thumbnail error: {e}", file=sys.stderr)
        return None


def detect_image_format(
    fp: Any,
    path_or_candidates: Union[str, Sequence[str], None] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> str:
    """Detects stereo format from image data and saves it to cache_db."""
    try:
        with PILImage.open(fp) as img:
            return (
                detect_stereo_format(
                    path_or_candidates,
                    image=img,
                    cache_db=cache_db,
                    cache_key=cache_key,
                    fallback_flat=True,
                )
                or FLAT
            )
    except Exception as e:
        print(f"Image format detection error: {e}", file=sys.stderr)
        return FLAT


_CUDA_HWACCEL: Optional[Any] = None
_CUDA_CHECKED: bool = False


def check_cuda_hwaccel_support() -> bool:
    """Checks if CUDA hardware acceleration is supported by PyAV and available on the system."""
    try:
        from av.codec.hwaccel import HWDevice, hwdevices_available

        if "cuda" not in hwdevices_available():
            return False
        # Create a device context to verify driver and device availability
        HWDevice("cuda")
        return True
    except Exception:
        return False


def init_video_hwaccel() -> bool:
    """Initializes CUDA hardware acceleration for video decoding if available."""
    global _CUDA_HWACCEL, _CUDA_CHECKED
    _CUDA_CHECKED = True
    if check_cuda_hwaccel_support():
        try:
            from av.codec.hwaccel import HWAccel

            _CUDA_HWACCEL = HWAccel(
                device_type="cuda",
                is_hw_owned=False,
                options={"primary_ctx": "1"},
                allow_software_fallback=True,
            )
            print("Video thumbnail hwaccel: CUDA enabled", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Video thumbnail hwaccel CUDA init failed: {e}", file=sys.stderr)
            _CUDA_HWACCEL = None
            return False
    else:
        _CUDA_HWACCEL = None
        return False


def get_video_hwaccel() -> Optional[Any]:
    """Gets the cached HWAccel instance, initializing if not yet checked."""
    global _CUDA_CHECKED
    if not _CUDA_CHECKED:
        init_video_hwaccel()
    return _CUDA_HWACCEL


_DECODE_LOCK = threading.Lock()


def _frame_to_thumbnail_image(frame: av.VideoFrame, max_size: int = 256) -> PILImage.Image:
    """Converts a video frame to a thumbnail-sized PIL Image using PyAV scaling and BT.709 color conversion."""
    reformat_kwargs: dict[str, Any] = {
        "dst_colorspace": "ITU709",
        "dst_color_primaries": "BT709",
        "dst_color_trc": "BT709",
    }
    if max(frame.width, frame.height) > max_size:
        scale = max_size / max(frame.width, frame.height)
        reformat_kwargs["width"] = max(2, int(frame.width * scale) // 2 * 2)
        reformat_kwargs["height"] = max(2, int(frame.height * scale) // 2 * 2)
    return frame.to_image(**reformat_kwargs)


def _decode_video_frame(fp: Any, hwaccel: Optional[Any] = None) -> Optional[PILImage.Image]:
    """Decodes a single representative video frame from the stream."""
    with _DECODE_LOCK:
        open_kwargs = {"mode": "r"}
        if hwaccel is not None:
            open_kwargs["hwaccel"] = hwaccel

        with av.open(fp, **open_kwargs) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.skip_frame = "NONKEY"

            duration_sec: Optional[float] = None
            if container.duration:
                duration_sec = float(container.duration / av.time_base)
            elif stream.duration is not None and stream.duration > 0 and stream.time_base is not None:
                duration_sec = float(stream.duration * stream.time_base)

            target_sec = compute_video_thumbnail_target_sec(duration_sec)
            container.seek(int(target_sec * av.time_base), backward=True, any_frame=False)
            for frame in container.decode(video=0):
                return _frame_to_thumbnail_image(frame)

            container.seek(0, backward=True, any_frame=False)
            for frame in container.decode(video=0):
                return _frame_to_thumbnail_image(frame)

        return None


def extract_video_frame(fp: Any) -> Optional[PILImage.Image]:
    """Extracts a representative video frame with CUDA hwaccel fallback."""
    hwaccel = get_video_hwaccel()
    try:
        return _decode_video_frame(fp, hwaccel=hwaccel)
    except Exception as e:
        if hwaccel is not None:
            try:
                if hasattr(fp, "seek"):
                    fp.seek(0)
                return _decode_video_frame(fp, hwaccel=None)
            except Exception as e_fallback:
                print(f"Video frame decode software fallback error: {e_fallback}", file=sys.stderr)
                return None
        print(f"Video frame decode error: {e}", file=sys.stderr)
        return None


def detect_video_format(
    fp: Any,
    path_or_candidates: Union[str, Sequence[str], None] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> str:
    """Detects stereo format from video frame and saves it to cache_db."""
    img = extract_video_frame(fp)
    if img is None:
        return FLAT
    return (
        detect_stereo_format(
            path_or_candidates,
            image=img,
            cache_db=cache_db,
            cache_key=cache_key,
            fallback_flat=True,
        )
        or FLAT
    )


def create_video_thumbnail(
    fp: Any,
    stereo_fmt: Optional[str] = None,
    path_or_candidates: Union[str, Sequence[str], None] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> Optional[bytes]:
    """Generates a 64x64 WebP thumbnail from a video stream."""
    img = extract_video_frame(fp)
    if img is None:
        return None
    return _resolve_and_render_thumbnail(
        img,
        stereo_fmt=stereo_fmt,
        path_or_candidates=path_or_candidates,
        cache_db=cache_db,
        cache_key=cache_key,
    )


def get_thumbnail_response(
    cache_db: Any,
    abs_path: Optional[str],
    zip_bytes: Optional[bytes],
    stereo_format_candidates: Sequence[str],
    etag: str,
    is_video: bool = False,
) -> Response:
    """Retrieves or generates a thumbnail HTTP response, using diskcache and ETag."""
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    thumb_data = cache_db.get(etag)
    if thumb_data is not None:
        return Response(
            content=thumb_data,
            media_type="image/webp",
            headers=headers,
        )

    fmt_cache_key = f"fmt_{etag}"
    candidates = stereo_format_candidates if stereo_format_candidates else ([abs_path] if abs_path else [])

    try:
        if zip_bytes is not None:
            thumb_data = create_thumbnail_data(
                io.BytesIO(zip_bytes),
                path_or_candidates=candidates,
                cache_db=cache_db,
                cache_key=fmt_cache_key,
            )
        elif abs_path and os.path.exists(abs_path):
            with open(abs_path, "rb") as f:
                creator = create_video_thumbnail if is_video else create_thumbnail_data
                thumb_data = creator(
                    f,
                    path_or_candidates=candidates,
                    cache_db=cache_db,
                    cache_key=fmt_cache_key,
                )
        else:
            raise HTTPException(status_code=404, detail="File not found")

        if thumb_data:
            cache_db.set(etag, thumb_data)
            return Response(
                content=thumb_data,
                media_type="image/webp",
                headers=headers,
            )
        return FileResponse(os.path.join(PUBLIC_DIR, "icons", "file-damage-fill.svg"))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Thumbnail generation error: {e}", file=sys.stderr)
        return FileResponse(os.path.join(PUBLIC_DIR, "icons", "file-damage-fill.svg"))
