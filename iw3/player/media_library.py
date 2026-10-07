import os
import io
import sys
import zipfile
import datetime
import traceback
import json
import av
import pysubs2
from PIL import Image as PILImage
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
from av.stream import Disposition
from typing import List, Dict, Any, Optional

from .server_state import ServerState
from .stereo_detector import detect_stereo_format, FLAT, SBS_FULL, SBS_HALF, SBS_FULL_CROSS, TB_FULL, TB_HALF
from .dir_config import PUBLIC_DIR


IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}
ARCHIVE_EXTENSIONS = {".zip"}
THUMBNAIL_SIZE = 64


def crop_left_eye(img: PILImage.Image, fmt: str) -> PILImage.Image:
    width, height = img.size
    if fmt in [SBS_FULL, SBS_HALF]:
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


def create_thumbnail_data(fp: io.BytesIO, stereo_fmt: str) -> Optional[bytes]:
    try:
        with PILImage.open(fp) as img:
            img = img.convert("RGBA")
            left_view = crop_left_eye(img, stereo_fmt)
            left_view.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), PILImage.Resampling.LANCZOS)
            canvas = PILImage.new("RGBA", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), (0, 0, 0, 0))
            offset = ((THUMBNAIL_SIZE - left_view.width) // 2, (THUMBNAIL_SIZE - left_view.height) // 2)
            canvas.paste(left_view, offset)
            output = io.BytesIO()
            canvas.save(output, format="WEBP", quality=80)
            return output.getvalue()
    except Exception as e:
        print(f"Thumbnail error: {e}", file=sys.stderr)
        return None


def create_video_thumbnail(fp: Any, stereo_fmt: str) -> Optional[bytes]:
    try:
        with av.open(fp, mode="r") as container:
            stream = container.streams.video[0]
            target_timestamp = int(1.0 / stream.time_base)
            container.seek(target_timestamp, any_frame=False, backward=True, stream=stream)
            for frame in container.decode(video=0):
                img = frame.to_image()
                img = img.convert("RGBA")
                left_view = crop_left_eye(img, stereo_fmt)
                left_view.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), PILImage.Resampling.LANCZOS)
                canvas = PILImage.new("RGBA", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), (0, 0, 0, 0))
                offset = ((THUMBNAIL_SIZE - left_view.width) // 2, (THUMBNAIL_SIZE - left_view.height) // 2)
                canvas.paste(left_view, offset)
                output = io.BytesIO()
                canvas.save(output, format="WEBP", quality=80)
                return output.getvalue()
    except Exception as e:
        print(f"Video thumbnail error: {e}", file=sys.stderr)
        return None


def extract_subtitle(fp: Any) -> Optional[List[Dict[str, str]]]:
    try:

        def get_priority_score(stream):
            score = 0
            if stream.disposition & Disposition.forced:
                score += 2
            if stream.disposition & Disposition.default:
                score += 1
            return score

        with av.open(fp, mode="r") as container:
            if len(container.streams.subtitles) == 0:
                return None

            subtitle_streams = list(container.streams.subtitles)
            subs_map = {}
            for i, stream in enumerate(subtitle_streams):
                lang = stream.metadata.get("language", "und")
                title = stream.metadata.get("title", None)
                display_title = f"{i:02d}:{lang}:{title}" if title else f"{i:02d}:{lang}"
                ssa_file = pysubs2.SSAFile()
                ssa_file.info["Title"] = display_title
                subs_map[stream.index] = ssa_file

            for packet in container.demux(subtitle_streams):
                if packet.stream.type == "subtitle":
                    if packet.pts is None or packet.duration is None:
                        continue
                    start_ms = int(float(packet.pts * packet.stream.time_base) * 1000)
                    duration_ms = int(float(packet.duration * packet.stream.time_base) * 1000)
                    end_ms = start_ms + duration_ms
                    for cue in packet.decode():
                        if not cue.ass:
                            continue
                        try:
                            text = cue.ass.decode("utf-8")
                        except UnicodeDecodeError:
                            text = cue.ass.decode("latin-1", errors="replace")
                        parts = text.split(",", 8)
                        actual_content = parts[8] if len(parts) > 8 else text
                        event = pysubs2.SSAEvent(start=start_ms, end=end_ms, text=actual_content.strip())
                        subs_map[packet.stream.index].append(event)

            results = []
            sorted_streams = sorted(subtitle_streams, key=get_priority_score, reverse=True)
            for stream in sorted_streams:
                ssa = subs_map[stream.index]
                if len(ssa) > 0:
                    results.append({"title": ssa.info["Title"], "vtt": ssa.to_string("vtt")})
            return results if results else None
    except Exception as e:
        print(f"Subtitle extraction error: {e}", file=sys.stderr)
        return None


class MediaLibrary:
    """Handles data logic for files, archives, thumbnails, and subtitles."""

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

    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
        items = []

        if internal_path is not None:
            abs_zip_path = self.safe_join(zip_rel)
            if not abs_zip_path or not os.path.exists(abs_zip_path):
                raise HTTPException(status_code=404, detail="ZIP file not found")
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
                            try:
                                dt = datetime.datetime(*info.date_time)
                                item_mtime = int(dt.timestamp())
                            except Exception:
                                item_mtime = zip_mtime
                            if is_dir:
                                if item_name not in seen_dirs:
                                    # Construct ZIP path using forward slashes only
                                    item_path = f"{zip_rel}/{prefix}{item_name}".replace("//", "/")
                                    items.append(
                                        {
                                            "path": item_path,
                                            "name": item_name,
                                            "type": "directory",
                                            "size": None,
                                            "mtime": item_mtime,
                                            "stereo_format": FLAT,
                                        }
                                    )
                                    seen_dirs.add(item_name)
                            else:
                                ext = os.path.splitext(item_name)[1].lower()
                                if ext in IMAGE_EXTENSIONS:
                                    # Construct ZIP path using forward slashes only
                                    item_path = f"{zip_rel}/{prefix}{item_name}".replace("//", "/")
                                    items.append(
                                        {
                                            "path": item_path,
                                            "name": item_name,
                                            "type": "image",
                                            "size": info.file_size,
                                            "mtime": item_mtime,
                                            "stereo_format": detect_stereo_format(
                                                [
                                                    item_name,
                                                    os.path.basename(prefix.rstrip("/")),
                                                    os.path.basename(zip_rel),
                                                ]
                                            ),
                                        }
                                    )
            except HTTPException:
                raise
            except Exception:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
            abs_path = self.safe_join(path)
            if not abs_path:
                raise HTTPException(status_code=403, detail="Access denied")
            if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
                raise HTTPException(status_code=404, detail="Directory not found")
            try:
                for entry in os.scandir(abs_path):
                    is_dir = entry.is_dir()
                    ext = os.path.splitext(entry.name)[1].lower()
                    item_type = None
                    if is_dir:
                        item_type = "directory"
                    elif ext in IMAGE_EXTENSIONS:
                        item_type = "image"
                    elif ext in VIDEO_EXTENSIONS:
                        item_type = "video"
                    elif ext in ARCHIVE_EXTENSIONS:
                        item_type = "archive"
                    if item_type:
                        stat = entry.stat()
                        items.append(
                            {
                                "path": os.path.relpath(entry.path, self.state.image_root).replace(os.sep, "/"),
                                "name": entry.name,
                                "type": item_type,
                                "size": stat.st_size if not is_dir else None,
                                "mtime": int(stat.st_mtime),
                                "stereo_format": detect_stereo_format(entry.path) if not is_dir else None,
                            }
                        )
            except Exception as e:
                print(f"List files error: {e}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Internal server error")
        return sorted(items, key=lambda x: (x["type"] != "directory", x["name"]))

    def get_file_response(self, path: str) -> Response:
        zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
        if internal_path:
            abs_zip_path = self.safe_join(zip_rel)
            if not abs_zip_path or not os.path.exists(abs_zip_path):
                raise HTTPException(status_code=404, detail="Archive not found")
            try:
                with zipfile.ZipFile(abs_zip_path, "r") as z:
                    with z.open(internal_path) as f:
                        data = f.read()
                        ext = os.path.splitext(internal_path)[1].lower()
                        mimetype = "image/jpeg"
                        if ext == ".png":
                            mimetype = "image/png"
                        elif ext == ".webp":
                            mimetype = "image/webp"
                        return Response(content=data, media_type=mimetype)
            except Exception as e:
                print(f"Archive extraction error: {e}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
            abs_path = self.safe_join(path)
            if not abs_path:
                raise HTTPException(status_code=403, detail="Access denied")
            return FileResponse(abs_path)

    def get_thumbnail(self, path: str, etag: str) -> Response:
        thumb_data = self.state.cache_db.get(etag)
        if thumb_data is not None:
            return Response(
                content=thumb_data,
                media_type="image/webp",
                headers={"ETag": etag, "Cache-Control": "no-cache"},
            )

        try:
            zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
            if internal_path:
                abs_zip_path = self.safe_join(zip_rel)
                with zipfile.ZipFile(abs_zip_path, "r") as z:
                    with z.open(internal_path) as f:
                        data_io = io.BytesIO(f.read())
                        fmt = detect_stereo_format(
                            [
                                os.path.basename(internal_path),
                                os.path.basename(os.path.dirname(internal_path)),
                                os.path.basename(zip_rel),
                            ]
                        )
                        thumb_data = create_thumbnail_data(data_io, fmt)
            else:
                abs_path = self.safe_join(path)
                if not abs_path or not os.path.exists(abs_path):
                    raise HTTPException(status_code=404, detail="File not found")
                fmt = detect_stereo_format(abs_path)
                ext = os.path.splitext(abs_path)[1].lower()
                with open(abs_path, "rb") as f:
                    thumb_data = create_video_thumbnail(f, fmt) if ext in VIDEO_EXTENSIONS else create_thumbnail_data(f, fmt)

            if thumb_data:
                self.state.cache_db.set(etag, thumb_data)
                return Response(
                    content=thumb_data,
                    media_type="image/webp",
                    headers={"ETag": etag, "Cache-Control": "no-cache"},
                )
            else:
                return FileResponse(os.path.join(PUBLIC_DIR, "icons", "file-damage-fill.svg"))
        except Exception as e:
            print(f"Thumbnail generation error: {e}", file=sys.stderr)
            return FileResponse(os.path.join(PUBLIC_DIR, "icons", "file-damage-fill.svg"))

    def get_subtitles(self, path: str, etag: str) -> Response:
        external_subs = []
        try:
            zip_rel, internal_path = self.split_zip_path(path.lstrip("/"))
            if not internal_path:
                abs_path = self.safe_join(path)
                if abs_path and os.path.exists(abs_path):
                    base_path = os.path.splitext(abs_path)[0]
                    vtt_path = base_path + ".vtt"
                    srt_path = base_path + ".srt"

                    target_ext_file = None
                    if os.path.exists(vtt_path):
                        target_ext_file = vtt_path
                    elif os.path.exists(srt_path):
                        target_ext_file = srt_path

                    if target_ext_file:
                        sub = pysubs2.load(target_ext_file)
                        external_subs.append({"title": "External Subtitle", "vtt": sub.to_string("vtt")})
        except Exception as e:
            print(f"External subtitle error: {e}", file=sys.stderr)

        # Get internal subtitles (from cache or extraction)
        vtt_data = self.state.cache_db.get(etag)
        if vtt_data is None:
            try:
                # Video extraction only for regular files
                abs_path = self.safe_join(path)
                if abs_path and os.path.exists(abs_path):
                    with open(abs_path, "rb") as f:
                        vtt_data = extract_subtitle(f)

                if vtt_data:
                    self.state.cache_db.set(etag, vtt_data)
            except Exception as e:
                print(f"Internal subtitle error: {e}", file=sys.stderr)
                vtt_data = []

        # Merge: External SRTs go first
        results = external_subs + (vtt_data or [])

        # Use no-cache to ensure clients always revalidate with ETag
        headers = {"ETag": etag, "Cache-Control": "no-cache"}

        return Response(
            content=json.dumps(results),
            media_type="application/json",
            headers=headers,
        )
