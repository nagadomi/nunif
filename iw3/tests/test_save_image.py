import argparse
import threading
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import torch
from PIL import Image

from iw3.utils import (
    bind_export_single_frame_callback,
    create_parser,
    iw3_main,
    process_video_keyframes,
    save_image,
    save_tensor_image,
    set_state_args,
)


def _create_args(input_path: str, output_path: str, *extra_args: str) -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args(
        [
            "-i",
            input_path,
            "-o",
            output_path,
            "--gpu",
            "-1",
            "--depth-model",
            "NULL",
            "--resolution",
            "32",
            "--method",
            "forward_fill",
            "--mapper",
            "none",
            "--format",
            "png",
            "-y",
            *extra_args,
        ]
    )
    set_state_args(args)
    return args


def _spy_pil_save(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    lock = threading.Lock()
    orig_save = Image.Image.save

    def wrapped_save(self: Image.Image, fp: Any, format: str | None = None, **params: Any) -> None:
        with lock:
            calls.append((str(fp), dict(params)))
        orig_save(self, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", wrapped_save)
    return calls


def _create_dummy_video(video_path: Path) -> None:
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=1)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 128, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_save_image_png_compress_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_pil_save(monkeypatch)
    im = Image.new("RGB", (32, 32), color=(120, 140, 160))
    tensor = torch.rand((3, 32, 32), dtype=torch.float32)

    default_path = tmp_path / "default.png"
    fast_path = tmp_path / "fast.png"
    tensor_default_path = tmp_path / "tensor_default.png"
    tensor_fast_path = tmp_path / "tensor_fast.png"

    save_image(im, str(default_path))
    save_image(im, str(fast_path), png_compress_level=1)
    save_tensor_image(tensor, str(tensor_default_path))
    save_tensor_image(tensor, str(tensor_fast_path), png_compress_level=1)

    recorded = {fp: params for fp, params in calls}
    assert recorded[str(default_path)]["compress_level"] == 6
    assert recorded[str(fast_path)]["compress_level"] == 1
    assert recorded[str(tensor_default_path)]["compress_level"] == 6
    assert recorded[str(tensor_fast_path)]["compress_level"] == 1


def test_save_image_workflow_png_compress_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    src_image = input_dir / "sample.png"
    Image.new("RGB", (32, 32), color=(64, 128, 192)).save(src_image)

    calls = _spy_pil_save(monkeypatch)

    # Single image conversion -> default compress_level 6
    single_out = tmp_path / "single_out.png"
    iw3_main(_create_args(str(src_image), str(single_out)))

    # Directory image conversion -> default compress_level 6
    batch_out_dir = tmp_path / "batch_out"
    iw3_main(_create_args(str(input_dir), str(batch_out_dir)))

    # Image export -> intermediate RGB uses compress_level 1
    export_dir = tmp_path / "export_out"
    iw3_main(_create_args(str(src_image), str(export_dir), "--export"))

    # Import from exported config -> final output uses compress_level 6
    import_out_dir = tmp_path / "import_out"
    iw3_main(_create_args(str(export_dir / "iw3_export.yml"), str(import_out_dir)))

    # Video export callback -> intermediate RGB uses compress_level 1
    vid_rgb_dir = tmp_path / "vid_export" / "rgb"
    vid_depth_dir = tmp_path / "vid_export" / "depth"
    vid_rgb_dir.mkdir(parents=True)
    vid_depth_dir.mkdir(parents=True)
    vid_args = _create_args("dummy.mp4", str(tmp_path / "vid_export"))
    vid_args.state["depth_model"].load(gpu=-1, resolution=32)
    with PoolExecutor(max_workers=2) as pool:
        cb = bind_export_single_frame_callback(
            vid_args.state["depth_model"],
            set(),
            str(vid_rgb_dir),
            str(vid_depth_dir),
            pool,
            vid_args,
        )
        frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 200, dtype=np.uint8), format="rgb24")
        frame.pts = 0
        cb(frame)
        cb(None)

    # Video keyframe conversion -> final output uses compress_level 6
    video_path = tmp_path / "sample.mp4"
    _create_dummy_video(video_path)
    keyframe_out_dir = tmp_path / "keyframe_out"
    keyframe_args = _create_args(str(video_path), str(keyframe_out_dir), "--keyframe")
    keyframe_args.state["depth_model"].load(gpu=-1, resolution=32)
    process_video_keyframes(
        str(video_path),
        str(keyframe_out_dir),
        keyframe_args,
        keyframe_args.state["depth_model"],
        None,
    )

    recorded = {fp: params for fp, params in calls}

    assert recorded[str(single_out)]["compress_level"] == 6

    batch_files = list(batch_out_dir.glob("*.png"))
    assert len(batch_files) == 1
    assert recorded[str(batch_files[0])]["compress_level"] == 6

    export_rgb_files = list((export_dir / "rgb").glob("*.png"))
    assert len(export_rgb_files) == 1
    assert recorded[str(export_rgb_files[0])]["compress_level"] == 1

    import_files = list(import_out_dir.glob("*.png"))
    assert len(import_files) == 1
    assert recorded[str(import_files[0])]["compress_level"] == 6

    vid_rgb_file = vid_rgb_dir / "00000000.png"
    assert vid_rgb_file.exists()
    assert recorded[str(vid_rgb_file)]["compress_level"] == 1

    keyframe_files = list((keyframe_out_dir / "sample_LRF_Full_SBS").glob("*.png"))
    assert len(keyframe_files) == 1
    assert recorded[str(keyframe_files[0])]["compress_level"] == 6
