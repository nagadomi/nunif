import pytest

from ..stereo_detector import (
    FLAT,
    SBS_180,
    SBS_FULL,
    SBS_FULL_CROSS,
    SBS_HALF,
    TB_FULL,
    TB_HALF,
    detect_stereo_format,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        # SBS Full
        ("video_Full_SBS.mp4", SBS_FULL),
        ("image_fullsbs.png", SBS_FULL),
        ("photo_LRF.jpg", SBS_FULL),
        ("movie_SBS.mp4", SBS_FULL),
        ("sample_3DHF.mkv", SBS_FULL),
        ("sample_3DPHF.mp4", SBS_FULL),
        # SBS Half
        ("video_LR.mp4", SBS_HALF),
        ("sample_3DH.mkv", SBS_HALF),
        ("clip_3DPH.webm", SBS_HALF),
        # SBS Full Cross-eyed
        ("image_RLF.png", SBS_FULL_CROSS),
        # VR180
        ("vr_LR_180.mp4", SBS_180),
        ("clip_180x180_LR.mp4", SBS_180),
        ("sample_VR180.mp4", SBS_180),
        ("VR180_scene.mp4", SBS_180),
        # TB Full
        ("movie_Full_TB.mp4", TB_FULL),
        ("video_fulltb.mkv", TB_FULL),
        ("photo_TBF.png", TB_FULL),
        ("clip_3DVF.mp4", TB_FULL),
        ("sample_3DPVF.webm", TB_FULL),
        # TB Half
        ("video_TB.mp4", TB_HALF),
        ("sample_3DV.mkv", TB_HALF),
        ("clip_3DPV.mp4", TB_HALF),
        # Untagged / Unrecognized (returns None unless fallback_flat=True)
        ("normal_photo.jpg", None),
        ("movie.mp4", None),
        ("archive.zip", None),
    ],
)
def test_detect_stereo_format_by_filename(filename, expected):
    assert detect_stereo_format(filename) == expected
    assert detect_stereo_format(filename, fallback_flat=True) == (expected or FLAT)


def test_detect_stereo_format_case_insensitive():
    assert detect_stereo_format("TEST_full_sbs.mp4") == SBS_FULL
    assert detect_stereo_format("image_lrf.png") == SBS_FULL
    assert detect_stereo_format("SAMPLE_3dh.MKV") == SBS_HALF
    assert detect_stereo_format("vr180_video.mp4") == SBS_180
    assert detect_stereo_format("clip_tb.mp4") == TB_HALF


def test_detect_stereo_format_parent_directory():
    # Filename has no tag, but parent directory has stereo tag
    assert detect_stereo_format("SBS_folder_3DH/video.mp4") == SBS_HALF
    assert detect_stereo_format("path/to/_Full_SBS/image.png") == SBS_FULL
    assert detect_stereo_format("my_VR180_collection/scene01.mp4") == SBS_180

    # Filename tag takes precedence over parent directory tag
    assert detect_stereo_format("path/to/_Full_TB/video_LR.mp4") == SBS_HALF


def test_detect_stereo_format_candidate_list():
    # Candidate list evaluation order (e.g. used for ZIP contents)
    assert detect_stereo_format(["photo_LRF.png", "inner_folder", "archive.zip"]) == SBS_FULL
    assert detect_stereo_format(["photo.png", "folder_3DH", "archive.zip"]) == SBS_HALF
    assert detect_stereo_format(["photo.png", "inner_folder", "archive_180x180_LR.zip"]) == SBS_180
    assert detect_stereo_format(["photo.png", "inner_folder", "archive.zip"]) is None
    assert detect_stereo_format(["photo.png", "inner_folder", "archive.zip"], fallback_flat=True) == FLAT


def _create_synthetic_test_images():
    import numpy as np
    from PIL import Image

    np.random.seed(42)

    # FLAT image: random pixels
    flat_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    flat_img = Image.fromarray(flat_arr)

    # SBS image: left and right halves are almost identical
    left = np.random.randint(20, 235, (64, 32, 3), dtype=np.uint8)
    noise_r = np.random.randint(-3, 4, (64, 32, 3))
    right = np.clip(left.astype(int) + noise_r, 0, 255).astype(np.uint8)
    sbs_arr = np.concatenate([left, right], axis=1)
    sbs_img = Image.fromarray(sbs_arr)

    # TB image: top and bottom halves are almost identical (16:9 aspect ratio for Half TB)
    top = np.random.randint(20, 235, (18, 64, 3), dtype=np.uint8)
    noise_b = np.random.randint(-3, 4, (18, 64, 3))
    bottom = np.clip(top.astype(int) + noise_b, 0, 255).astype(np.uint8)
    tb_arr = np.concatenate([top, bottom], axis=0)
    tb_img = Image.fromarray(tb_arr)

    return flat_img, sbs_img, tb_img


def test_predict_labels():
    from ..stereo_detector import predict

    flat_img, sbs_img, tb_img = _create_synthetic_test_images()
    assert predict(flat_img) == "FLAT"
    assert predict(sbs_img) == "SBS"
    assert predict(tb_img) == "TB"


def test_detect_stereo_format_with_image():
    flat_img, sbs_img, tb_img = _create_synthetic_test_images()

    # When filename has no tags and no VR/180 keywords:
    assert detect_stereo_format("photo.jpg", image=flat_img) == FLAT
    assert detect_stereo_format("photo.jpg", image=sbs_img) == SBS_HALF
    assert detect_stereo_format("photo.jpg", image=tb_img) == TB_HALF

    # When filename contains 'VR' or '180':
    assert detect_stereo_format("vr_vacation.jpg", image=sbs_img) == SBS_180
    assert detect_stereo_format("beach_180.mp4", image=sbs_img) == SBS_180
    # TB format always resolves to TB_HALF (VR180 is SBS only)
    assert detect_stereo_format("my_180_video.mp4", image=tb_img) == TB_HALF
    assert detect_stereo_format("VR_scene.mp4", image=tb_img) == TB_HALF
    # Flat image with VR/180 in name remains FLAT
    assert detect_stereo_format("vr_recording.mp4", image=flat_img) == FLAT

    # Candidate list / parent directory with 'vr' or '180'
    assert detect_stereo_format(["photo.jpg", "vr_folder"], image=sbs_img) == SBS_180
    assert detect_stereo_format("vr_folder/photo.jpg", image=sbs_img) == SBS_180

    # Tag precedence: Explicit tag in filename overrides image prediction
    assert detect_stereo_format("photo_Full_SBS.jpg", image=flat_img) == SBS_FULL
    assert detect_stereo_format("photo_LRF.jpg", image=tb_img) == SBS_FULL


def test_detect_stereo_format_caching():
    from ..stereo_detector import compute_format_cache_key

    # Cache key computation test
    key_with_mtime = compute_format_cache_key("/path/to/file.jpg", 1234567890)
    assert key_with_mtime.startswith("fmt_")
    key_no_mtime = compute_format_cache_key("/path/to/archive.zip/inner.jpg")
    assert key_no_mtime.startswith("fmt_")

    # Simple dict-based mock cache DB
    cache_db = {}
    _, sbs_img, _ = _create_synthetic_test_images()

    test_path = "vacation_180.mp4"
    cache_key = compute_format_cache_key(test_path, 1000)

    # First call: not cached, predict and store into cache_db
    res1 = detect_stereo_format(test_path, image=sbs_img, cache_db=cache_db, cache_key=cache_key)
    assert res1 == SBS_180
    assert cache_key in cache_db
    assert cache_db[cache_key] == SBS_180

    # Second call: cached, should return cached result without even needing image
    res2 = detect_stereo_format(test_path, image=None, cache_db=cache_db, cache_key=cache_key)
    assert res2 == SBS_180

    # Second call with dummy image: cache still takes precedence
    dummy_flat = _create_synthetic_test_images()[0]
    res3 = detect_stereo_format(test_path, image=dummy_flat, cache_db=cache_db, cache_key=cache_key)
    assert res3 == SBS_180


def test_extract_candidate_names():
    from ..stereo_detector import extract_candidate_names

    assert extract_candidate_names(None) == []
    assert extract_candidate_names("") == []
    assert extract_candidate_names("/path/to/my_folder/video.mp4") == ["video.mp4", "my_folder"]
    assert extract_candidate_names(["video.mp4", "archive.zip"]) == ["video.mp4", "archive.zip"]
    assert extract_candidate_names(["/path/to/folder/video.mp4"]) == ["video.mp4", "folder"]


def test_detect_format_from_tags():
    from ..stereo_detector import detect_format_from_tags

    assert detect_format_from_tags("video_Full_SBS.mp4") == SBS_FULL
    assert detect_format_from_tags("my_180x180_LR/photo.jpg") == SBS_180
    assert detect_format_from_tags("no_tags_photo.jpg") is None


def test_detect_format_from_image():
    from ..stereo_detector import detect_format_from_image

    flat_img, sbs_img, tb_img = _create_synthetic_test_images()

    assert detect_format_from_image(flat_img, "any_name.jpg") == FLAT
    assert detect_format_from_image(sbs_img, "normal_name.jpg") == SBS_HALF
    assert detect_format_from_image(sbs_img, "vr_scene.jpg") == SBS_180
    assert detect_format_from_image(sbs_img, ["scene.jpg", "vr_dir"]) == SBS_180
    assert detect_format_from_image(tb_img, "any_name.jpg") == TB_HALF


def test_detect_format_from_image_aspect_ratio():
    import numpy as np
    from PIL import Image

    from ..stereo_detector import detect_format_from_image

    def make_sbs_image(w: int, h: int) -> Image.Image:
        np.random.seed(123)
        mid = w // 2
        left = np.random.randint(20, 235, (h, mid, 3), dtype=np.uint8)
        noise = np.random.randint(-2, 3, (h, mid, 3))
        right = np.clip(left.astype(int) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np.concatenate([left, right], axis=1))

    # Half SBS: 16:9 or narrower
    img_16_9 = make_sbs_image(160, 90)  # 16:9 = 1.777...
    assert detect_format_from_image(img_16_9, "video.mp4") == SBS_HALF

    img_4_3 = make_sbs_image(120, 90)  # 4:3 = 1.333...
    assert detect_format_from_image(img_4_3, "video.mp4") == SBS_HALF

    img_854_480 = make_sbs_image(854, 480)  # 16:9 standard resolution approx
    assert detect_format_from_image(img_854_480, "video.mp4") == SBS_HALF

    # Full SBS: wider than 16:9
    img_2_1 = make_sbs_image(200, 100)  # 2:1 = 2.0
    assert detect_format_from_image(img_2_1, "video.mp4") == SBS_FULL

    img_8_3 = make_sbs_image(240, 90)  # 8:3 = 2.666...
    assert detect_format_from_image(img_8_3, "video.mp4") == SBS_FULL

    img_32_9 = make_sbs_image(320, 90)  # 32:9 = 3.555...
    assert detect_format_from_image(img_32_9, "video.mp4") == SBS_FULL

    # VR180 has precedence over Full SBS even if wide
    assert detect_format_from_image(img_2_1, "my_vr_scene.mp4") == SBS_180

    def make_tb_image(w: int, h: int) -> Image.Image:
        np.random.seed(123)
        mid = h // 2
        top = np.random.randint(20, 235, (mid, w, 3), dtype=np.uint8)
        noise = np.random.randint(-2, 3, (mid, w, 3))
        bottom = np.clip(top.astype(int) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np.concatenate([top, bottom], axis=0))

    # Half TB: 16:9 or 4:3 (horizontal frames)
    img_tb_16_9 = make_tb_image(160, 90)  # 16:9 = 1.777...
    assert detect_format_from_image(img_tb_16_9, "video.mp4") == TB_HALF

    img_tb_4_3 = make_tb_image(120, 90)  # 4:3 = 1.333...
    assert detect_format_from_image(img_tb_4_3, "video.mp4") == TB_HALF

    # Full TB: 1:1 (VR Over-Under), 8:9 (16:9 Full), 2:3 (4:3 Full)
    img_tb_1_1 = make_tb_image(100, 100)  # 1:1 = 1.0 (VR Over-Under)
    assert detect_format_from_image(img_tb_1_1, "video.mp4") == TB_FULL

    img_tb_8_9 = make_tb_image(80, 90)  # 8:9 = 0.888...
    assert detect_format_from_image(img_tb_8_9, "video.mp4") == TB_FULL

    img_tb_2_3 = make_tb_image(60, 90)  # 2:3 = 0.666...
    assert detect_format_from_image(img_tb_2_3, "video.mp4") == TB_FULL


def test_predict_uniform_and_blank_images():
    from PIL import Image

    from ..stereo_detector import predict

    black = Image.new("RGB", (64, 64), (0, 0, 0))
    white = Image.new("RGB", (64, 64), (255, 255, 255))
    red = Image.new("RGB", (64, 64), (255, 0, 0))
    tiny = Image.new("RGB", (1, 1), (128, 128, 128))

    assert predict(black) == "FLAT"
    assert predict(white) == "FLAT"
    assert predict(red) == "FLAT"
    assert predict(tiny) == "FLAT"


def test_extract_features_and_predict_modes():
    import numpy as np
    from PIL import Image

    from ..stereo_detector import extract_features, predict

    # RGBA mode
    rgba_arr = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
    feat_rgba = extract_features(rgba_img)
    assert feat_rgba.shape == (4,)
    assert predict(rgba_img) in ["FLAT", "SBS", "TB"]

    # Grayscale ("L") mode
    l_arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    l_img = Image.fromarray(l_arr, mode="L")
    feat_l = extract_features(l_img)
    assert feat_l.shape == (4,)
    assert predict(l_img) in ["FLAT", "SBS", "TB"]

    # Palette ("P") mode
    p_img = rgba_img.convert("P")
    feat_p = extract_features(p_img)
    assert feat_p.shape == (4,)
    assert predict(p_img) in ["FLAT", "SBS", "TB"]


def test_compute_format_cache_key_leading_slash():
    from ..stereo_detector import compute_format_cache_key

    # Should produce identical keys regardless of leading slash
    key1 = compute_format_cache_key("/path/to/media.jpg", 12345)
    key2 = compute_format_cache_key("path/to/media.jpg", 12345)
    assert key1 == key2

    key3 = compute_format_cache_key("/archive.zip/pic.jpg")
    key4 = compute_format_cache_key("archive.zip/pic.jpg")
    assert key3 == key4
