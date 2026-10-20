import os
import zipfile
from types import SimpleNamespace

import pytest
from diskcache import Cache
from fastapi import HTTPException
from PIL import Image as PILImage

from ..media_library import MediaLibrary
from ..stereo_detector import SBS_FULL, SBS_HALF, TB_HALF


@pytest.fixture
def mock_server_state(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    state = SimpleNamespace(
        image_root=str(tmp_path / "media"),
        auth_user=None,
        auth_password=None,
        debug_mode=False,
        debug_console=False,
        cache_db=Cache(str(cache_dir)),
        secret_key_file=str(tmp_path / "secret.key"),
    )
    os.makedirs(state.image_root, exist_ok=True)
    yield state
    state.cache_db.close()


def test_split_zip_path(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    assert lib.split_zip_path("") == (None, None)
    assert lib.split_zip_path("images/photo.jpg") == ("images/photo.jpg", None)
    assert lib.split_zip_path("archives/set.zip") == ("archives/set.zip", "")
    assert lib.split_zip_path("archives/set.zip/inner/01.jpg") == ("archives/set.zip", "inner/01.jpg")


def test_safe_join_traversal(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    root = os.path.abspath(mock_server_state.image_root)

    # Valid subpaths (with or without leading slash)
    assert lib.safe_join("") == root
    assert lib.safe_join("photos/pic.jpg") == os.path.join(root, "photos/pic.jpg")
    assert lib.safe_join("/photos/pic.jpg") == os.path.join(root, "photos/pic.jpg")

    # Directory traversal attempts should return None
    assert lib.safe_join("../secret.txt") is None
    assert lib.safe_join("/../etc/passwd") is None
    assert lib.safe_join("../../etc/passwd") is None
    assert lib.safe_join("photos/../../etc/passwd") is None


def test_list_files_filesystem(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    # Create dummy directory structure
    os.makedirs(os.path.join(media_root, "subfolder"), exist_ok=True)
    with open(os.path.join(media_root, "photo_LRF.png"), "wb") as f:
        f.write(b"dummy image data")
    with open(os.path.join(media_root, "video_LR.mp4"), "wb") as f:
        f.write(b"dummy video data")
    with open(os.path.join(media_root, "archive.zip"), "wb") as f:
        f.write(b"dummy zip data")
    with open(os.path.join(media_root, "ignored.txt"), "wb") as f:
        f.write(b"ignore this file")

    items = lib.list_files("/")
    names = [item["name"] for item in items]

    # Directories must come first, followed by alphabetical files
    assert names[0] == "subfolder"
    assert "photo_LRF.png" in names
    assert "video_LR.mp4" in names
    assert "archive.zip" in names
    assert "ignored.txt" not in names  # Non-media files should be excluded

    # Verify detected format
    photo_item = next(it for it in items if it["name"] == "photo_LRF.png")
    assert photo_item["type"] == "image"
    assert photo_item["stereo_format"] == SBS_FULL

    video_item = next(it for it in items if it["name"] == "video_LR.mp4")
    assert video_item["type"] == "video"
    assert video_item["stereo_format"] == SBS_HALF


def test_list_files_zip_archive(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    zip_path = os.path.join(media_root, "album_SBS.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("folder1/pic1.jpg", b"image content")
        z.writestr("folder1/pic2_TB.png", b"image content")
        z.writestr("root_pic.webp", b"image content")

    # List root of zip
    items = lib.list_files("album_SBS.zip")
    names = [item["name"] for item in items]
    assert "folder1" in names
    assert "root_pic.webp" in names

    # Check stereo format detection in zip
    # Inherits from zip name 'album_SBS.zip' -> SBS_FULL
    root_pic = next(it for it in items if it["name"] == "root_pic.webp")
    assert root_pic["stereo_format"] == SBS_FULL

    # List subfolder inside zip
    sub_items = lib.list_files("album_SBS.zip/folder1")
    sub_names = [it["name"] for it in sub_items]
    assert "pic1.jpg" in sub_names
    assert "pic2_TB.png" in sub_names

    pic2 = next(it for it in sub_items if it["name"] == "pic2_TB.png")
    assert pic2["stereo_format"] == TB_HALF


def test_list_files_not_found(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    with pytest.raises(HTTPException) as excinfo:
        lib.list_files("nonexistent_folder")
    assert excinfo.value.status_code == 404


def test_list_files_format_cache_coordination(mock_server_state):
    import numpy as np

    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    # Create synthetic SBS image without stereo tags, but containing '180' in filename
    left = np.random.randint(20, 235, (64, 32, 3), dtype=np.uint8)
    noise_r = np.random.randint(-3, 4, (64, 32, 3))
    right = np.clip(left.astype(int) + noise_r, 0, 255).astype(np.uint8)
    sbs_arr = np.concatenate([left, right], axis=1)
    sbs_img = PILImage.fromarray(sbs_arr)

    filename = "beach_180.png"
    file_path = os.path.join(media_root, filename)
    sbs_img.save(file_path, format="PNG")

    # Initial list_files -> no tag, no cache yet -> None
    items = lib.list_files("/")
    item = next(it for it in items if it["name"] == filename)
    assert item["stereo_format"] is None

    # Request thumbnail (generates thumbnail & caches format prediction)
    resp = lib.get_thumbnail(filename)
    assert resp.status_code == 200
    etag = resp.headers.get("etag")
    assert lib.get_thumbnail(filename, if_none_match=etag).status_code == 304

    # Re-call list_files -> now format is retrieved from cache -> SBS_180
    items_after = lib.list_files("/")
    item_after = next(it for it in items_after if it["name"] == filename)
    assert item_after["stereo_format"] == "SBS_180"


def test_get_file_info(mock_server_state):
    import numpy as np

    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    # Tagged file
    tagged_img = PILImage.new("RGB", (200, 100), color="blue")
    tagged_path = os.path.join(media_root, "movie_Full_SBS.png")
    tagged_img.save(tagged_path, format="PNG")

    info = lib.get_file_info("movie_Full_SBS.png")
    assert info["name"] == "movie_Full_SBS.png"
    assert info["type"] == "image"
    assert info["stereo_format"] == SBS_FULL

    # Untagged image (trigger prediction via get_file_info)
    left = np.random.randint(20, 235, (64, 32, 3), dtype=np.uint8)
    noise_r = np.random.randint(-3, 4, (64, 32, 3))
    right = np.clip(left.astype(int) + noise_r, 0, 255).astype(np.uint8)
    sbs_arr = np.concatenate([left, right], axis=1)
    sbs_img = PILImage.fromarray(sbs_arr)
    untagged_path = os.path.join(media_root, "scene_180.png")
    sbs_img.save(untagged_path, format="PNG")

    # Initial list_files has None
    items = lib.list_files("/")
    scene_item = next(it for it in items if it["name"] == "scene_180.png")
    assert scene_item["stereo_format"] is None

    # get_file_info predicts and returns SBS_180
    info_pred = lib.get_file_info("scene_180.png")
    assert info_pred["name"] == "scene_180.png"
    assert info_pred["type"] == "image"
    assert info_pred["stereo_format"] == "SBS_180"

    # Subsequent list_files returns cached SBS_180
    items_after = lib.list_files("/")
    scene_after = next(it for it in items_after if it["name"] == "scene_180.png")
    assert scene_after["stereo_format"] == "SBS_180"

    # Directory
    sub_dir = os.path.join(media_root, "subfolder")
    os.makedirs(sub_dir, exist_ok=True)
    dir_info = lib.get_file_info("subfolder")
    assert dir_info["type"] == "directory"
    assert dir_info["stereo_format"] is None

    # Nonexistent file
    with pytest.raises(HTTPException) as excinfo:
        lib.get_file_info("nonexistent.png")
    assert excinfo.value.status_code == 404


def test_list_files_broken_symlinks(mock_server_state):
    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    # Create valid target files and directory
    valid_file = os.path.join(media_root, "valid.png")
    with open(valid_file, "wb") as f:
        f.write(b"valid image")
    valid_dir = os.path.join(media_root, "valid_folder")
    os.makedirs(valid_dir, exist_ok=True)

    # Create broken symlinks
    broken_file_link = os.path.join(media_root, "broken_file.png")
    os.symlink(os.path.join(media_root, "nonexistent.png"), broken_file_link)

    broken_dir_link = os.path.join(media_root, "broken_dir")
    os.symlink(os.path.join(media_root, "nonexistent_dir"), broken_dir_link)

    # Create valid symlinks
    valid_file_link = os.path.join(media_root, "valid_symlink.png")
    os.symlink(valid_file, valid_file_link)

    valid_dir_link = os.path.join(media_root, "valid_dir_symlink")
    os.symlink(valid_dir, valid_dir_link)

    items = lib.list_files("/")
    names = [item["name"] for item in items]

    # Broken symlinks should be skipped without raising exceptions
    assert "broken_file.png" not in names
    assert "broken_dir" not in names

    # Valid targets and valid symlinks should remain
    assert "valid_folder" in names
    assert "valid_dir_symlink" in names
    assert "valid.png" in names
    assert "valid_symlink.png" in names


def test_zip_archive_media_and_thumbnail(mock_server_state):
    import io

    lib = MediaLibrary(mock_server_state)
    media_root = mock_server_state.image_root

    img = PILImage.new("RGB", (64, 64), color="green")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    png_bytes = bio.getvalue()

    zip_path = os.path.join(media_root, "photos.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("inner/pic.png", png_bytes)

    # Valid entry in archive
    media_resp = lib.get_file_response("photos.zip/inner/pic.png")
    assert media_resp.status_code == 200
    assert media_resp.media_type == "image/png"
    assert media_resp.body == png_bytes

    # Missing entry in archive -> 404
    with pytest.raises(HTTPException) as excinfo:
        lib.get_file_response("photos.zip/inner/missing.png")
    assert excinfo.value.status_code == 404

    # Thumbnail generation and ETag 304 for archive entry
    thumb_resp = lib.get_thumbnail("photos.zip/inner/pic.png")
    assert thumb_resp.status_code == 200
    etag = thumb_resp.headers.get("etag")
    assert etag is not None
    assert lib.get_thumbnail("photos.zip/inner/pic.png", if_none_match=etag).status_code == 304
