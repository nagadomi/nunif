import datetime
import hashlib
import io
import os
import sys
import traceback
import zipfile
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from .server_state import ServerState
from .stereo_detector import compute_format_cache_key, compute_media_etag, detect_stereo_format
from .subtitle_service import find_external_subtitle_path, get_subtitles_response
from .thumbnail_service import (
    VIDEO_EXTENSIONS,
    detect_image_format,
    detect_video_format,
    get_thumbnail_response,
)

__all__ = ["MediaLibrary"]

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".webp"}
ARCHIVE_EXTENSIONS = {".zip"}
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class MediaLibrary:
    """Handles data logic for files, directory traversal, archives, and media streaming."""

    def __init__(self, state: ServerState):
        self.state = state

    def split_zip_path(self, rel_path: str):
        if not rel_path:
            return None, None
        rel_path = rel_path.replace(os.sep, "/")
        parts = rel_path.split("/")
        for i, part in enumerate(parts):
            if part.lower().endswith(".zip"):
                zip_rel = "/".join(parts[: i + 1])
                internal = "/".join(parts[i + 1 :])
                return zip_rel, internal
        return rel_path, None

    def safe_join(self, rel_path: str) -> Optional[str]:
        if not rel_path:
            return self.state.image_root
        abs_root = os.path.abspath(self.state.image_root)
        target_path = os.path.join(abs_root, rel_path.lstrip("/"))
        abs_path = os.path.abspath(target_path)
        try:
            if os.path.commonpath([abs_root, abs_path]) == abs_root:
                return abs_path
        except ValueError:
            return None
        return None

    def _resolve_fs_path(self, rel_path: str, must_be_dir: bool = False) -> str:
        abs_path = self.safe_join(rel_path)
        if not abs_path:
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.exists(abs_path) or (must_be_dir and not os.path.isdir(abs_path)):
            detail = "Directory not found" if must_be_dir else "File not found"
            raise HTTPException(status_code=404, detail=detail)
        return abs_path

    def _resolve_zip_path(self, zip_rel: str) -> str:
        abs_zip_path = self.safe_join(zip_rel)
        if not abs_zip_path:
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.isfile(abs_zip_path):
            raise HTTPException(status_code=404, detail="ZIP file not found")
        return abs_zip_path

    @staticmethod
    def _get_item_type(name: str, is_dir: bool = False) -> Optional[str]:
        """Determines item type from name and directory status."""
        if is_dir:
            return "directory"
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        if ext in ARCHIVE_EXTENSIONS:
            return "archive"
        return None

    @staticmethod
    def _parse_zip_mtime(info: zipfile.ZipInfo, fallback_mtime: int) -> int:
        """Parses modified timestamp from ZipInfo with fallback."""
        try:
            dt = datetime.datetime(*info.date_time)
            return int(dt.timestamp())
        except Exception:
            return fallback_mtime

    def _resolve_zip_stereo_format(
        self,
        z: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        item_name: str,
        item_path: str,
        parent_dir: str,
        zip_rel: str,
        predict_if_unknown: bool = False,
    ) -> Optional[str]:
        """Resolves stereo format for a zip entry from tags, cache, or optional image prediction."""
        cache_key = compute_format_cache_key(item_path)
        candidates = [
            item_name,
            os.path.basename(parent_dir.rstrip("/")),
            os.path.basename(zip_rel),
        ]
        stereo_fmt = detect_stereo_format(
            candidates,
            cache_db=self.state.cache_db,
            cache_key=cache_key,
        )
        if stereo_fmt is None and predict_if_unknown:
            with z.open(info) as f:
                data = f.read()
                stereo_fmt = detect_image_format(
                    io.BytesIO(data),
                    path_or_candidates=candidates,
                    cache_db=self.state.cache_db,
                    cache_key=cache_key,
                )
        return stereo_fmt

    def _build_fs_item(
        self,
        abs_path: str,
        stat: Optional[os.stat_result] = None,
        is_dir: Optional[bool] = None,
        predict_if_unknown: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Constructs a MediaItem dictionary for a filesystem entry."""
        try:
            if is_dir is None:
                is_dir = os.path.isdir(abs_path)
            item_name = os.path.basename(abs_path)
            item_type = self._get_item_type(item_name, is_dir=is_dir)
            if not item_type:
                return None

            if stat is None:
                stat = os.stat(abs_path)
            item_rel = os.path.relpath(abs_path, self.state.image_root).replace(os.sep, "/")

            stereo_fmt = None
            if not is_dir:
                cache_key = compute_format_cache_key(item_rel, int(stat.st_mtime))
                stereo_fmt = detect_stereo_format(
                    [abs_path],
                    cache_db=self.state.cache_db,
                    cache_key=cache_key,
                )
                if stereo_fmt is None and predict_if_unknown:
                    if item_type == "video":
                        with open(abs_path, "rb") as f:
                            stereo_fmt = detect_video_format(
                                f,
                                path_or_candidates=[abs_path],
                                cache_db=self.state.cache_db,
                                cache_key=cache_key,
                            )
                    elif item_type == "image":
                        with open(abs_path, "rb") as f:
                            stereo_fmt = detect_image_format(
                                f,
                                path_or_candidates=[abs_path],
                                cache_db=self.state.cache_db,
                                cache_key=cache_key,
                            )

            return {
                "path": item_rel,
                "name": item_name,
                "type": item_type,
                "size": stat.st_size if not is_dir else None,
                "mtime": int(stat.st_mtime),
                "stereo_format": stereo_fmt,
            }
        except OSError:
            return None

    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
        items = []

        if internal_path is not None:
            abs_zip_path = self._resolve_zip_path(zip_rel)
            try:
                zip_mtime = int(os.path.getmtime(abs_zip_path))
                with zipfile.ZipFile(abs_zip_path, "r") as z:
                    prefix = internal_path.strip("/")
                    if prefix:
                        prefix += "/"
                    seen_dirs = set()
                    for info in z.infolist():
                        name = info.filename
                        if name.startswith(prefix) and name != prefix:
                            rel_to_prefix = name[len(prefix) :]
                            parts = rel_to_prefix.split("/")
                            item_name = parts[0]
                            is_dir = len(parts) > 1 or name.endswith("/")
                            item_mtime = self._parse_zip_mtime(info, zip_mtime)
                            if is_dir:
                                if item_name not in seen_dirs:
                                    item_path = f"{zip_rel}/{prefix}{item_name}".replace("//", "/")
                                    items.append(
                                        {
                                            "path": item_path,
                                            "name": item_name,
                                            "type": "directory",
                                            "size": None,
                                            "mtime": item_mtime,
                                            "stereo_format": None,
                                        }
                                    )
                                    seen_dirs.add(item_name)
                            else:
                                ext = os.path.splitext(item_name)[1].lower()
                                if ext in IMAGE_EXTENSIONS:
                                    item_path = f"{zip_rel}/{prefix}{item_name}".replace("//", "/")
                                    stereo_fmt = self._resolve_zip_stereo_format(
                                        z,
                                        info,
                                        item_name=item_name,
                                        item_path=item_path,
                                        parent_dir=prefix,
                                        zip_rel=zip_rel,
                                        predict_if_unknown=False,
                                    )
                                    items.append(
                                        {
                                            "path": item_path,
                                            "name": item_name,
                                            "type": "image",
                                            "size": info.file_size,
                                            "mtime": item_mtime,
                                            "stereo_format": stereo_fmt,
                                        }
                                    )
            except HTTPException:
                raise
            except Exception:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
            abs_path = self._resolve_fs_path(path, must_be_dir=True)
            try:
                for entry in os.scandir(abs_path):
                    try:
                        stat = entry.stat()
                        is_dir = entry.is_dir()
                    except OSError:
                        # Skip broken symlinks or inaccessible entries
                        continue
                    item = self._build_fs_item(
                        entry.path,
                        stat=stat,
                        is_dir=is_dir,
                        predict_if_unknown=False,
                    )
                    if item:
                        items.append(item)
            except Exception as e:
                print(f"List files error: {e}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Internal server error")
        return sorted(items, key=lambda x: (x["type"] != "directory", x["name"]))

    def get_file_response(self, path: str) -> Response:
        zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
        if internal_path:
            abs_zip_path = self._resolve_zip_path(zip_rel)
            try:
                with zipfile.ZipFile(abs_zip_path, "r") as z:
                    with z.open(internal_path) as f:
                        data = f.read()
                        ext = os.path.splitext(internal_path)[1].lower()
                        mimetype = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
                        return Response(content=data, media_type=mimetype)
            except KeyError:
                raise HTTPException(status_code=404, detail="File not found in archive")
            except Exception as e:
                print(f"Archive extraction error: {e}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
            abs_path = self._resolve_fs_path(path)
            if not os.path.isfile(abs_path):
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(abs_path)

    def get_thumbnail(
        self,
        path: str,
        etag: Optional[str] = None,
        if_none_match: Optional[str] = None,
    ) -> Response:
        norm_path = path.lstrip("/")
        zip_rel, internal_path = self.split_zip_path(norm_path)

        abs_path: Optional[str] = None
        if not internal_path:
            abs_path = self.safe_join(norm_path)

        if etag is None:
            mtime = int(os.path.getmtime(abs_path)) if (abs_path and os.path.exists(abs_path)) else None
            etag = compute_media_etag(norm_path, mtime)

        if if_none_match and if_none_match == etag:
            return Response(status_code=304)

        if internal_path:
            abs_zip_path = self._resolve_zip_path(zip_rel)
            zip_bytes: Optional[bytes] = None
            if self.state.cache_db.get(etag) is None:
                try:
                    with zipfile.ZipFile(abs_zip_path, "r") as z:
                        with z.open(internal_path) as f:
                            zip_bytes = f.read()
                except KeyError:
                    raise HTTPException(status_code=404, detail="File not found in archive")
                except Exception as e:
                    print(f"Archive read error: {e}", file=sys.stderr)
                    raise HTTPException(status_code=500, detail="Internal server error")

            candidates = [
                os.path.basename(internal_path),
                os.path.basename(os.path.dirname(internal_path)),
                os.path.basename(zip_rel),
            ]
            return get_thumbnail_response(
                cache_db=self.state.cache_db,
                abs_path=None,
                zip_bytes=zip_bytes,
                stereo_format_candidates=candidates,
                etag=etag,
                is_video=False,
            )
        else:
            abs_path = self._resolve_fs_path(norm_path)
            ext = os.path.splitext(abs_path)[1].lower()
            return get_thumbnail_response(
                cache_db=self.state.cache_db,
                abs_path=abs_path,
                zip_bytes=None,
                stereo_format_candidates=[abs_path],
                etag=etag,
                is_video=ext in VIDEO_EXTENSIONS,
            )

    def get_subtitles(
        self,
        path: str,
        etag: Optional[str] = None,
        if_none_match: Optional[str] = None,
    ) -> Response:
        norm_path = path.lstrip("/")
        zip_rel, internal_path = self.split_zip_path(norm_path)
        if internal_path:
            # Archives do not support video, so no subtitles
            return Response(content="[]", media_type="application/json")

        abs_path = self.safe_join(norm_path)
        video_mtime = int(os.path.getmtime(abs_path)) if (abs_path and os.path.exists(abs_path)) else None
        internal_cache_key = f"subs_internal_{compute_media_etag(norm_path, video_mtime)}"

        if etag is None:
            raw_key = f"subs_{norm_path}"
            if abs_path and video_mtime is not None:
                raw_key += f"{video_mtime}"
                ext_sub = find_external_subtitle_path(abs_path)
                if ext_sub:
                    ext_name = os.path.splitext(ext_sub)[1].lstrip(".")
                    raw_key += f"_{ext_name}_{int(os.path.getmtime(ext_sub))}"
            etag = hashlib.sha256(raw_key.encode()).hexdigest()

        if if_none_match and if_none_match == etag:
            return Response(status_code=304)

        return get_subtitles_response(
            cache_db=self.state.cache_db,
            abs_path=abs_path,
            etag=etag,
            internal_cache_key=internal_cache_key,
        )

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Returns metadata and verified stereo format for an individual file or directory."""
        norm_path = path.lstrip("/")
        zip_rel, internal_path = self.split_zip_path(norm_path)

        if internal_path is not None:
            abs_zip_path = self._resolve_zip_path(zip_rel)
            try:
                zip_mtime = int(os.path.getmtime(abs_zip_path))
                with zipfile.ZipFile(abs_zip_path, "r") as z:
                    clean_internal = internal_path.strip("/")
                    matched_info = None
                    is_dir = False
                    for info in z.infolist():
                        fname = info.filename.rstrip("/")
                        if fname == clean_internal:
                            matched_info = info
                            is_dir = info.filename.endswith("/")
                            break
                    if not matched_info:
                        raise HTTPException(status_code=404, detail="File not found in archive")

                    item_mtime = self._parse_zip_mtime(matched_info, zip_mtime)
                    item_name = os.path.basename(clean_internal)
                    item_path = f"{zip_rel}/{clean_internal}".replace("//", "/")

                    if is_dir:
                        return {
                            "path": item_path,
                            "name": item_name,
                            "type": "directory",
                            "size": None,
                            "mtime": item_mtime,
                            "stereo_format": None,
                        }

                    ext = os.path.splitext(item_name)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        raise HTTPException(status_code=400, detail="Unsupported archive file type")

                    stereo_fmt = self._resolve_zip_stereo_format(
                        z,
                        matched_info,
                        item_name=item_name,
                        item_path=item_path,
                        parent_dir=os.path.dirname(clean_internal),
                        zip_rel=zip_rel,
                        predict_if_unknown=True,
                    )
                    return {
                        "path": item_path,
                        "name": item_name,
                        "type": "image",
                        "size": matched_info.file_size,
                        "mtime": item_mtime,
                        "stereo_format": stereo_fmt,
                    }
            except HTTPException:
                raise
            except Exception as e:
                print(f"Archive get_file_info error: {e}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
            abs_path = self._resolve_fs_path(path)
            item = self._build_fs_item(abs_path, predict_if_unknown=True)
            if not item:
                raise HTTPException(status_code=400, detail="Unsupported media type")
            return item
