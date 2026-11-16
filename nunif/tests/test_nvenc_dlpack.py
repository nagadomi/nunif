import io
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
import torch
from av.video.frame import CudaContext

from nunif.utils.video import VideoOutputConfig, generate_video, has_nvenc, process_video
from nunif.utils.video.color_transform import (
    ColorPrimaries,
    ColorRange,
    Colorspace,
    ColorTransform,
    ColorTrc,
    OutputTransform,
)


def _nvenc_ready() -> bool:
    return torch.cuda.is_available() and has_nvenc()


def _make_quadrant_pattern(width: int, height: int, device: torch.device | str) -> torch.Tensor:
    img = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    hh, hw = height // 2, width // 2
    img[0, :hh, :hw] = 1.0  # Red
    img[1, :hh, hw:] = 1.0  # Green
    img[2, hh:, :hw] = 1.0  # Blue
    img[:, hh:, hw:] = 1.0  # White
    return img


def _assert_quadrant_colors(rgb: np.ndarray, width: int, height: int) -> None:
    hh, hw = height // 2, width // 2
    tl = rgb[hh // 2, hw // 2]
    tr = rgb[hh // 2, hw + hw // 2]
    bl = rgb[hh + hh // 2, hw // 2]
    br = rgb[hh + hh // 2, hw + hw // 2]
    assert tl[0] > 200 and tl[1] < 50 and tl[2] < 50
    assert tr[1] > 200 and tr[0] < 50 and tr[2] < 50
    assert bl[2] > 200 and bl[0] < 50 and bl[1] < 50
    assert br[0] > 200 and br[1] > 200 and br[2] > 200


def _create_sample_mp4(path: Path, width: int = 642, height: int = 362, frames: int = 8) -> str:
    with av.open(str(path), "w") as container:
        vs = cast(av.VideoStream, container.add_stream("libx264", rate=30))
        vs.width = width
        vs.height = height
        vs.pix_fmt = "yuv420p"
        aus = cast(av.AudioStream, container.add_stream("aac", rate=48000))
        for i in range(frames):
            vf = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), dtype=np.uint8), format="rgb24")
            vf.pts = i
            for pkt in vs.encode(vf):
                container.mux(pkt)
            af = av.AudioFrame.from_ndarray(np.zeros((2, 1024), dtype=np.float32), format="fltp", layout="stereo")
            af.sample_rate = 48000
            af.pts = i * 1024
            for pkt in aus.encode(af):
                container.mux(pkt)
        for pkt in vs.encode(None):
            container.mux(pkt)
        for pkt in aus.encode(None):
            container.mux(pkt)
    return str(path)


@pytest.mark.parametrize("out_format,itemsize", [("nv12", 1), ("p010le", 2)])
@pytest.mark.parametrize("width,height", [(640, 360), (642, 362)])
def test_to_yuv_planes_contiguous_aligned(out_format: str, itemsize: int, width: int, height: int) -> None:
    y = torch.full((1, 1, height, width), 0.5, dtype=torch.float32)
    u = torch.full((1, 1, height // 2, width // 2), 0.25, dtype=torch.float32)
    v = torch.full((1, 1, height // 2, width // 2), 0.75, dtype=torch.float32)

    y_p, uv_p = ColorTransform.to_yuv_planes(y, u, v, out_format=out_format)

    assert y_p.shape == (height, width)
    assert uv_p.shape == (height // 2, width)
    assert y_p.stride(0) == uv_p.stride(0)
    linesize_bytes = y_p.stride(0) * itemsize
    assert linesize_bytes % 512 == 0
    assert linesize_bytes >= width * itemsize
    assert uv_p.data_ptr() - y_p.data_ptr() == height * linesize_bytes


@pytest.mark.hwaccel
@pytest.mark.skipif(not _nvenc_ready(), reason="CUDA and NVENC required")
@pytest.mark.parametrize(
    "codec,pix_fmt,width,height",
    [
        ("h264_nvenc", "nv12", 640, 360),
        ("h264_nvenc", "nv12", 642, 362),
        ("hevc_nvenc", "nv12", 642, 362),
        ("hevc_nvenc", "p010le", 642, 362),
        ("hevc_nvenc", "nv12", 1920, 1080),
    ],
)
def test_nvenc_dlpack_direct_roundtrip(codec: str, pix_fmt: str, width: int, height: int) -> None:
    device = torch.device("cuda:0")
    buf = io.BytesIO()
    buf.name = "test.mp4"
    container = av.open(buf, mode="w", format="mp4")
    stream = cast(av.VideoStream, container.add_stream(codec, rate=30))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "cuda"

    cuda_ctx = CudaContext(device_id=0, current_ctx=True, primary_ctx=False)
    ot = OutputTransform(
        dst_pix_fmt=pix_fmt,
        dst_colorspace=Colorspace.ITU709,
        dst_color_primaries=ColorPrimaries.BT709,
        dst_color_trc=ColorTrc.BT709,
        dst_color_range=ColorRange.MPEG,
        cuda_context=cuda_ctx,
    )

    for i in range(4):
        frame = ot(_make_quadrant_pattern(width, height, device))
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)

    assert stream.codec_context.pix_fmt == "cuda"
    assert stream.codec_context.sw_format is not None
    assert stream.codec_context.sw_format.name == pix_fmt
    container.close()

    buf.seek(0)
    with av.open(buf, mode="r") as dec:
        for df in dec.decode(video=0):
            _assert_quadrant_colors(df.to_ndarray(format="rgb24"), width, height)
            break
        else:
            pytest.fail("No decoded frames produced")


@pytest.mark.hwaccel
@pytest.mark.skipif(not _nvenc_ready(), reason="CUDA and NVENC required")
@pytest.mark.parametrize(
    "codec,pix_fmt,out_wh",
    [
        ("h264_nvenc", "yuv420p", None),
        ("hevc_nvenc", "yuv420p", (642, 362)),
        ("hevc_nvenc", "yuv420p10le", (642, 362)),
    ],
)
def test_generate_video_nvenc(
    tmp_path: Path,
    codec: str,
    pix_fmt: str,
    out_wh: tuple[int, int] | None,
) -> None:
    device = torch.device("cuda:0")
    width, height = 642, 362
    audio_mp4 = _create_sample_mp4(tmp_path / "audio_src.mp4", width=width, height=height)
    out_mp4 = str(tmp_path / "out.mp4")

    def gen():
        for _ in range(5):
            yield _make_quadrant_pattern(width, height, device)

    cfg = VideoOutputConfig(
        fps=30,
        pix_fmt=pix_fmt,
        video_codec=codec,
        output_width=out_wh[0] if out_wh else None,
        output_height=out_wh[1] if out_wh else None,
    )
    generate_video(out_mp4, gen, cfg, audio_file=audio_mp4, device=device)

    with av.open(out_mp4, mode="r") as dec:
        assert len(dec.streams.audio) == 1
        for df in dec.decode(video=0):
            _assert_quadrant_colors(df.to_ndarray(format="rgb24"), width, height)
            break
        else:
            pytest.fail("No decoded frames produced")


@pytest.mark.hwaccel
@pytest.mark.skipif(not _nvenc_ready(), reason="CUDA and NVENC required")
def test_process_video_nvenc(tmp_path: Path) -> None:
    device = torch.device("cuda:0")
    width, height = 642, 362
    in_mp4 = _create_sample_mp4(tmp_path / "in.mp4", width=width, height=height)
    out_mp4 = str(tmp_path / "out.mp4")

    def cb(frame: Any) -> torch.Tensor | None:
        if frame is None:
            return None
        return _make_quadrant_pattern(frame.width, frame.height, device)

    process_video(
        in_mp4,
        out_mp4,
        cb,
        config_callback=lambda _m: VideoOutputConfig(fps=30, pix_fmt="yuv420p", video_codec="hevc_nvenc"),
        device=device,
    )

    with av.open(out_mp4, mode="r") as dec:
        assert len(dec.streams.audio) == 1
        for df in dec.decode(video=0):
            _assert_quadrant_colors(df.to_ndarray(format="rgb24"), width, height)
            break
        else:
            pytest.fail("No decoded frames produced")
