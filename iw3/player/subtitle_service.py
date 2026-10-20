import json
import os
import sys
from typing import Any, Dict, List, Optional
import av
from av.stream import Disposition
from fastapi import Response
import pysubs2


def extract_subtitle(fp: Any) -> Optional[List[Dict[str, str]]]:
    """Extracts internal subtitles from a video container using PyAV."""
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


def find_external_subtitle_path(abs_path: str) -> Optional[str]:
    """Finds an associated external .vtt or .srt subtitle file if it exists."""
    base_path = os.path.splitext(abs_path)[0]
    for ext in (".vtt", ".srt"):
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    return None


def get_external_subtitles(abs_path: str) -> List[Dict[str, str]]:
    """Loads external subtitle files (.vtt or .srt) associated with a media file."""
    external_subs = []
    try:
        target_ext_file = find_external_subtitle_path(abs_path)
        if target_ext_file:
            sub = pysubs2.load(target_ext_file)
            external_subs.append({"title": "External Subtitle", "vtt": sub.to_string("vtt")})
    except Exception as e:
        print(f"External subtitle error: {e}", file=sys.stderr)
    return external_subs


def get_subtitles_response(
    cache_db: Any,
    abs_path: Optional[str],
    etag: str,
    internal_cache_key: Optional[str] = None,
) -> Response:
    """Retrieves external and internal subtitles, merges them, and returns a JSON response."""
    external_subs = []
    if abs_path and os.path.exists(abs_path):
        external_subs = get_external_subtitles(abs_path)

    cache_key = internal_cache_key or etag
    vtt_data = cache_db.get(cache_key)
    if vtt_data is None:
        try:
            if abs_path and os.path.exists(abs_path):
                with open(abs_path, "rb") as f:
                    extracted = extract_subtitle(f)
                vtt_data = extracted if extracted is not None else []
                cache_db.set(cache_key, vtt_data)
            else:
                vtt_data = []
        except Exception as e:
            print(f"Internal subtitle error: {e}", file=sys.stderr)
            vtt_data = []

    results = external_subs + (vtt_data or [])
    headers = {"ETag": etag, "Cache-Control": "no-cache"}

    return Response(
        content=json.dumps(results),
        media_type="application/json",
        headers=headers,
    )
