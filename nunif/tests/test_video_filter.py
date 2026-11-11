from fractions import Fraction
from pathlib import Path

import av
import pytest
import torch
from av.video.reformatter import Colorspace, ColorTrc

import nunif.utils.video.video_preprocessor as video_preprocessor
from nunif.utils.video.color_transform import ColorRange, TensorFrame
from nunif.utils.video.video_filter.av_filter_graph import AVFilterGraph, _escape_ffmpeg_filter_option
from nunif.utils.video.video_filter.tensor_filter_graph import TensorFilterGraph

_IDENTITY_CUBE_2X2X2 = "\n".join(
    [
        "LUT_3D_SIZE 2",
        "0.0 0.0 0.0",
        "1.0 0.0 0.0",
        "0.0 1.0 0.0",
        "1.0 1.0 0.0",
        "0.0 0.0 1.0",
        "1.0 0.0 1.0",
        "0.0 1.0 1.0",
        "1.0 1.0 1.0",
    ]
)


def _create_test_cube(base_dir: Path, subdir: str, filename: str) -> str:
    cube_dir = base_dir / subdir
    cube_dir.mkdir(parents=True, exist_ok=True)
    cube_path = cube_dir / filename
    cube_path.write_text(_IDENTITY_CUBE_2X2X2 + "\n", encoding="utf-8")
    return str(cube_path)


def test_av_filter_graph_lut3d_raw_path_with_special_chars(tmp_path: Path) -> None:
    cube_path = _create_test_cube(tmp_path, "sub dir", "a,b=c'd.cube")
    filters = AVFilterGraph.parse_vf_option(f"lut3d={cube_path.replace(',', r'\,')}")

    graph = av.filter.Graph()
    AVFilterGraph.build_graph(
        graph,
        pix_fmt="rgb24",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        video_filters=filters,
    )

    frame = av.VideoFrame(16, 16, "rgb24")
    graph.push(frame)
    out = graph.pull()
    assert isinstance(out, av.VideoFrame)
    assert (out.width, out.height) == (16, 16)


def test_av_filter_graph_lut3d_preescaped_options_not_double_escaped(tmp_path: Path) -> None:
    cube_path = _create_test_cube(tmp_path, "sub dir", "pre=escaped.cube")
    escaped_path = _escape_ffmpeg_filter_option(cube_path)
    filters = [("lut3d", f"file={escaped_path}:interp=tetrahedral")]

    graph = av.filter.Graph()
    AVFilterGraph.build_graph(
        graph,
        pix_fmt="rgb24",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        video_filters=filters,
    )

    frame = av.VideoFrame(16, 16, "rgb24")
    graph.push(frame)
    out = graph.pull()
    assert isinstance(out, av.VideoFrame)


def test_update_hdr2sdr_video_filter_works_with_both_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube_path = _create_test_cube(tmp_path, "dir,with=special", "lut'file.cube")
    monkeypatch.setattr(video_preprocessor, "_get_lut_path", lambda **_kwargs: cube_path)

    vf = video_preprocessor._update_hdr2sdr_video_filter(
        vf="",
        colorspace=Colorspace.BT2020,
        color_trc=ColorTrc.SMPTE2084,
        output_colorspace="bt709-tv",
    )

    av_filters = AVFilterGraph.parse_vf_option(vf)
    graph = av.filter.Graph()
    AVFilterGraph.build_graph(
        graph,
        pix_fmt="rgb24",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        video_filters=av_filters,
    )
    av_frame = av.VideoFrame(16, 16, "rgb24")
    graph.push(av_frame)
    assert isinstance(graph.pull(), av.VideoFrame)

    tensor_graph = TensorFilterGraph(vf)
    t_frame = TensorFrame(
        planes=torch.zeros((1, 3, 16, 16)),
        pts=0,
        dts=0,
        time_base=None,
        colorspace=Colorspace.BT2020,
        color_primaries=9,
        color_trc=ColorTrc.SMPTE2084,
        color_range=ColorRange.MPEG,
        side_data=None,
        use_16bit=False,
    )
    out_t = tensor_graph.update(t_frame)
    assert out_t is not None
    assert out_t.planes.shape == (1, 3, 16, 16)
