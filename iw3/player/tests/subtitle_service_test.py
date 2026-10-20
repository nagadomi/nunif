from ..subtitle_service import find_external_subtitle_path, get_external_subtitles, get_subtitles_response


def test_get_external_subtitles_vtt(tmp_path):
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy video")

    vtt_file = tmp_path / "video.vtt"
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello World!
"""
    vtt_file.write_text(vtt_content, encoding="utf-8")

    assert find_external_subtitle_path(str(media_file)) == str(vtt_file)
    subs = get_external_subtitles(str(media_file))
    assert len(subs) == 1
    assert subs[0]["title"] == "External Subtitle"
    assert "Hello World!" in subs[0]["vtt"]


def test_get_external_subtitles_none_found(tmp_path):
    media_file = tmp_path / "video_no_subs.mp4"
    media_file.write_bytes(b"dummy video")

    assert find_external_subtitle_path(str(media_file)) is None
    subs = get_external_subtitles(str(media_file))
    assert subs == []


def test_get_external_subtitles_srt(tmp_path):
    media_file = tmp_path / "movie.mp4"
    media_file.write_bytes(b"dummy video")

    srt_file = tmp_path / "movie.srt"
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Testing SRT subtitle
"""
    srt_file.write_text(srt_content, encoding="utf-8")

    subs = get_external_subtitles(str(media_file))
    assert len(subs) == 1
    assert subs[0]["title"] == "External Subtitle"
    # pysubs2 converts srt to vtt format
    assert "Testing SRT subtitle" in subs[0]["vtt"]


def test_get_subtitles_response_internal_cache_key(tmp_path):
    class DummyCache(dict):
        def set(self, key, value):
            self[key] = value

    cache_db = DummyCache()
    cache_db["subs_internal_key"] = [
        {"title": "00:eng", "vtt": "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nCached internal"}
    ]

    media_file = tmp_path / "movie.mp4"
    media_file.write_bytes(b"dummy video")
    srt_file = tmp_path / "movie.srt"
    srt_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nExternal line\n", encoding="utf-8")

    resp = get_subtitles_response(
        cache_db=cache_db,
        abs_path=str(media_file),
        etag="response_etag_v1",
        internal_cache_key="subs_internal_key",
    )
    assert resp.status_code == 200
    assert resp.headers.get("ETag") == "response_etag_v1"
    body = resp.body.decode("utf-8")
    assert "External Subtitle" in body
    assert "Cached internal" in body
