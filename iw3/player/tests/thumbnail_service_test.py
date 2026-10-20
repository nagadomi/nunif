import io

from PIL import Image as PILImage

from ..stereo_detector import FLAT, SBS_180, SBS_FULL, SBS_FULL_CROSS, SBS_HALF, TB_FULL, TB_HALF
from ..thumbnail_service import (
    THUMBNAIL_SIZE,
    VIDEO_THUMBNAIL_MAX_SEC,
    VIDEO_THUMBNAIL_MIN_SEC,
    compute_video_thumbnail_target_sec,
    create_thumbnail_data,
    crop_left_eye,
    render_thumbnail,
)


def test_crop_left_eye_all_formats():
    img_wide = PILImage.new("RGB", (200, 100))
    assert crop_left_eye(img_wide, SBS_FULL).size == (100, 100)
    assert crop_left_eye(img_wide, SBS_HALF).size == (200, 100)
    assert crop_left_eye(img_wide, SBS_180).size == (100, 100)
    assert crop_left_eye(img_wide, SBS_FULL_CROSS).size == (100, 100)

    img_tall = PILImage.new("RGB", (100, 200))
    assert crop_left_eye(img_tall, TB_FULL).size == (100, 100)
    assert crop_left_eye(img_tall, TB_HALF).size == (100, 200)

    img_flat = PILImage.new("RGB", (150, 150))
    assert crop_left_eye(img_flat, FLAT).size == (150, 150)


def test_render_thumbnail():
    img = PILImage.new("RGB", (200, 100), color="red")
    thumb = render_thumbnail(img, SBS_FULL)
    assert thumb is not None
    loaded = PILImage.open(io.BytesIO(thumb))
    assert loaded.format == "WEBP"
    assert loaded.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)


def test_compute_video_thumbnail_target_sec():
    # When duration is None or 0, fallback to min sec
    assert compute_video_thumbnail_target_sec(None) == VIDEO_THUMBNAIL_MIN_SEC
    assert compute_video_thumbnail_target_sec(0) == VIDEO_THUMBNAIL_MIN_SEC

    # When duration is <= min sec, scale proportionally
    assert compute_video_thumbnail_target_sec(2.0) == 2.0 * 0.20
    assert compute_video_thumbnail_target_sec(5.0) == 5.0 * 0.20

    # Normal duration (clamped by min sec)
    assert compute_video_thumbnail_target_sec(10.0) == 5.0  # 10 * 0.2 = 2.0 -> clamped to 5.0

    # Normal duration (between min and max)
    assert compute_video_thumbnail_target_sec(100.0) == 20.0  # 100 * 0.2 = 20.0

    # Long duration (clamped by max sec)
    assert compute_video_thumbnail_target_sec(1000.0) == VIDEO_THUMBNAIL_MAX_SEC


def test_create_thumbnail_data_webp_output():
    img = PILImage.new("RGBA", (300, 150), color="blue")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)

    thumb = create_thumbnail_data(bio, SBS_FULL)
    assert thumb is not None
    loaded = PILImage.open(io.BytesIO(thumb))
    assert loaded.format == "WEBP"
    assert loaded.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)


def test_create_thumbnail_data_invalid_input():
    invalid_io = io.BytesIO(b"not an image data")
    assert create_thumbnail_data(invalid_io, SBS_FULL) is None


def _create_dummy_video_stream(num_frames: int = 15, width: int = 128, height: int = 64) -> io.BytesIO:
    """Generates an in-memory MP4 video stream for testing."""
    import av

    bio = io.BytesIO()
    container = av.open(bio, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=24)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        img = PILImage.new("RGB", (width, height), color=(i * 15 % 255, 120, 200))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    bio.seek(0)
    return bio


def test_create_video_thumbnail_with_generated_video():
    from ..thumbnail_service import check_cuda_hwaccel_support, create_video_thumbnail

    video_io = _create_dummy_video_stream(num_frames=20)
    thumb = create_video_thumbnail(video_io, SBS_FULL)
    assert thumb is not None
    loaded = PILImage.open(io.BytesIO(thumb))
    assert loaded.format == "WEBP"
    assert loaded.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)

    # Calling check_cuda_hwaccel_support should return a boolean without crashing
    assert isinstance(check_cuda_hwaccel_support(), bool)


def test_create_thumbnail_data_image_detection_fallback():
    import numpy as np

    # Create SBS image without stereo tags in name
    left = np.random.randint(20, 235, (64, 32, 3), dtype=np.uint8)
    noise_r = np.random.randint(-3, 4, (64, 32, 3))
    right = np.clip(left.astype(int) + noise_r, 0, 255).astype(np.uint8)
    sbs_arr = np.concatenate([left, right], axis=1)
    sbs_img = PILImage.fromarray(sbs_arr)

    bio = io.BytesIO()
    sbs_img.save(bio, format="PNG")
    bio.seek(0)

    # Candidates contain '180'
    thumb = create_thumbnail_data(bio, FLAT, path_or_candidates=["vacation_180.jpg"])
    assert thumb is not None
    loaded = PILImage.open(io.BytesIO(thumb))
    assert loaded.format == "WEBP"
    assert loaded.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)


def test_get_thumbnail_response_cache_hit():
    from ..thumbnail_service import get_thumbnail_response

    class DummyCache(dict):
        def set(self, key, value):
            self[key] = value

    cache_db = DummyCache()
    cache_db["dummy_etag"] = b"dummy_webp_bytes"

    resp = get_thumbnail_response(
        cache_db=cache_db,
        abs_path=None,
        zip_bytes=None,
        stereo_format_candidates=[],
        etag="dummy_etag",
    )
    assert resp.status_code == 200
    assert resp.body == b"dummy_webp_bytes"
    assert resp.headers.get("ETag") == "dummy_etag"
