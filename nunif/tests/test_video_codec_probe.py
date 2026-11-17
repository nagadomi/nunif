import av
import pytest

from iw3.desktop import video_stream_encoder as VSE
from nunif.utils.video import utils as VU


def test_has_video_codec_caches_probe_results(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    probe_calls: list[tuple[str, int | None]] = []

    def fake_probe(codec: str, gpu_id: int | None = None) -> bool:
        probe_calls.append((codec, gpu_id))
        return codec == "h264_qsv"

    monkeypatch.setattr(VU, "_probe_video_codec", fake_probe)
    monkeypatch.setattr(VU.av.codec, "codecs_available", {"h264_qsv", "h264_amf"})

    assert VU.has_video_codec("h264_qsv") is True
    assert VU.has_video_codec("h264_qsv") is True
    assert VU.has_video_codec("h264_amf") is False
    assert VU.has_video_codec("h264_amf") is False
    assert probe_calls == [("h264_qsv", None), ("h264_amf", None)]


def test_has_video_codec_software_skips_encode_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()

    def fail_if_called(codec: str, gpu_id: int | None = None) -> bool:
        raise AssertionError(f"Software codec {codec} should not run dummy encode probe")

    monkeypatch.setattr(VU, "_probe_video_codec", fail_if_called)
    monkeypatch.setattr(
        VU.av.codec,
        "codecs_available",
        {"libx264", "libopenh264", "libx265", "libvpx-vp9", "utvideo", "ffv1"},
    )
    assert VU.has_video_codec("libx264") is True
    assert VU.has_video_codec("libopenh264") is True
    assert VU.has_video_codec("libx265") is True
    assert VU.has_video_codec("libvpx-vp9") is True
    assert VU.has_vp9() is True
    assert VU.has_video_codec("utvideo") is True
    assert VU.has_video_codec("ffv1") is True


def test_has_video_codec_respects_codecs_available_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    monkeypatch.setattr(VU, "_probe_video_codec", lambda codec, gpu_id=None: True)
    monkeypatch.setattr(VU.av.codec, "codecs_available", {"h264_qsv"})
    assert VU.has_video_codec("h264_qsv") is True

    monkeypatch.setattr(VU.av.codec, "codecs_available", set())
    assert VU.has_video_codec("h264_qsv") is False


def test_has_video_codec_vaapi_skips_encode_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()

    def fail_if_called(codec: str, gpu_id: int | None = None) -> bool:
        raise AssertionError("VAAPI should not run dummy encode probe")

    monkeypatch.setattr(VU, "_probe_video_codec", fail_if_called)
    monkeypatch.setattr(VU.av.codec, "codecs_available", {"h264_vaapi", "hevc_vaapi"})
    monkeypatch.setattr(VU, "hwdevices_available", lambda: ["vaapi"])
    assert VU.has_video_codec("h264_vaapi") is True
    assert VU.has_vaapi() is True

    monkeypatch.setattr(VU, "hwdevices_available", lambda: [])
    assert VU.has_video_codec("h264_vaapi") is False
    assert VU.has_vaapi() is False


def test_has_video_codec_nvenc_checks_gpus_0_to_3(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    probe_calls: list[tuple[str, int | None]] = []

    def fake_probe(codec: str, gpu_id: int | None = None) -> bool:
        probe_calls.append((codec, gpu_id))
        return gpu_id == 2

    # Should still probe NVENC even when PyTorch reports ROCm or no CUDA device.
    monkeypatch.setattr(VU, "IS_ROCM", True)
    monkeypatch.setattr(VU, "_probe_video_codec", fake_probe)
    monkeypatch.setattr(VU.av.codec, "codecs_available", {"h264_nvenc"})
    monkeypatch.setattr(VU.torch.cuda, "is_available", lambda: False)

    assert VU.has_video_codec("h264_nvenc") is True
    assert probe_calls == [
        ("h264_nvenc", 0),
        ("h264_nvenc", 1),
        ("h264_nvenc", 2),
    ]
    assert VU.has_video_codec("h264_nvenc", gpu_id=0) is False
    assert VU.has_video_codec("h264_nvenc", gpu_id=2) is True
    assert len(probe_calls) == 3


def test_stream_video_codecs_does_not_skip_nvenc_on_non_cuda_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    monkeypatch.setattr(VU, "IS_ROCM", True)
    monkeypatch.setattr(VSE.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(VSE.av.codec, "codecs_available", {"libx264", "h264_nvenc"})
    monkeypatch.setattr(
        VU,
        "_probe_video_codec",
        lambda codec, gpu_id=None: codec == "libx264" or (codec == "h264_nvenc" and gpu_id == 0),
    )

    available = VSE.get_available_stream_video_codecs()
    assert "h264_nvenc" in available
    assert "libx264" in available


def test_has_video_codec_real_probe_behavior() -> None:
    VU.clear_video_codec_cache()
    assert VU.has_video_codec("nonexistent_video_codec") is False

    if "libx264" in av.codec.codecs_available:
        assert VU.has_video_codec("libx264") is True

    if "h264_nvenc" in av.codec.codecs_available:
        assert VU.has_video_codec("h264_nvenc", gpu_id=99) is False


def test_stream_video_codecs_filter_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    monkeypatch.setattr(
        VSE.av.codec,
        "codecs_available",
        {"libx264", "h264_qsv", "h264_amf", "h264_vaapi"},
    )
    monkeypatch.setattr(VSE, "HW_DEVICES", ["vaapi"])
    monkeypatch.setattr(VU, "hwdevices_available", lambda: ["vaapi"])
    monkeypatch.setattr(
        VU,
        "_probe_video_codec",
        lambda codec, gpu_id=None: codec == "libx264",
    )

    available = VSE.get_available_stream_video_codecs()
    assert "libx264" in available
    assert "h264_vaapi" in available
    assert "h264_qsv" not in available
    assert "h264_amf" not in available
    assert "jpeg" in available

    assert VSE.validate_stream_video_codec("libx264") == "libx264"
    assert VSE.validate_stream_video_codec("h264_vaapi") == "h264_vaapi"
    with pytest.raises(ValueError):
        VSE.validate_stream_video_codec("h264_amf")


def test_missing_vaapi_and_libopenh264_do_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    VU.clear_video_codec_cache()
    monkeypatch.setattr(VSE.av.codec, "codecs_available", {"libx264"})
    monkeypatch.setattr(VSE, "HW_DEVICES", [])
    monkeypatch.setattr(VU, "hwdevices_available", lambda: [])
    monkeypatch.setattr(VU, "_probe_video_codec", lambda codec, gpu_id=None: codec == "libx264")

    assert VU.has_video_codec("libopenh264") is False
    assert VU.has_video_codec("h264_vaapi") is False
    assert VU.has_vaapi() is False
    available = VSE.get_available_stream_video_codecs()
    assert "libopenh264" not in available
    assert "h264_vaapi" not in available
    assert "libx264" in available
    assert "jpeg" in available

    with pytest.raises(ValueError):
        VSE.validate_stream_video_codec("libopenh264")
    with pytest.raises(ValueError):
        VSE.validate_stream_video_codec("h264_vaapi")
