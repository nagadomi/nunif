import os
from typing import NamedTuple

import av
import numpy as np
import pytest
from PIL import Image

from nunif.utils import video as VU
from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data


class CliSetup(NamedTuple):
    image: str
    image_rgba: str
    video: str
    video_rgba: str
    video_rgba_vp9: str | None
    video_hdr: str
    dir: str
    output: str
    cli: list[str]


@pytest.fixture(scope="module")
def cli_setup(tmp_path_factory: pytest.TempPathFactory) -> CliSetup:
    tmp_dir = tmp_path_factory.mktemp("cli")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    return CliSetup(
        image=os.path.join(smoke_dir, "sd.png"),
        image_rgba=os.path.join(smoke_dir, "dog_448_rgba.png"),
        video=os.path.join(smoke_dir, "sd.mkv"),
        video_rgba=os.path.join(smoke_dir, "rgba.mkv"),
        video_rgba_vp9=os.path.join(smoke_dir, "rgba_vp9.mkv") if VU.has_vp9() else None,
        video_hdr=os.path.join(smoke_dir, "hdr.mkv"),
        dir=smoke_dir,
        output=output_dir,
        cli=["python", "-m", "iw3.cli", "-y", "-o", output_dir],
    )


def test_cli_image(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.image, "--depth-model", "Any_S", "--metadata"])


def test_cli_video(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--metadata"])


def test_cli_dir(cli_setup: CliSetup) -> None:
    run(
        [
            "python",
            "-m",
            "iw3",
            "-y",
            "-i",
            cli_setup.dir,
            "-o",
            cli_setup.output,
            "--depth-model",
            "Any_S",
            "--metadata",
        ]
    )


def test_cli_ema(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
        ]
    )


def test_cli_batch(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "Any_S",
            "--metadata",
            "--batch-size",
            "4",
            "--max-workers",
            "2",
            "--cuda-stream",
        ]
    )


def test_cli_low_vram(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--metadata", "--low-vram"])


def test_cli_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "VDA_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--disable-scene-cache",
        ]
    )


def test_cli_vda_stream(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "VDA_Stream_S",
            "--metadata",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )


def test_cli_hdr(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
            "--depth-model",
            "Any_S",
            "--colorspace",
            "auto",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p10le",
        ]
    )


def test_cli_hdr_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
            "--depth-model",
            "VDA_S",
            "--colorspace",
            "auto",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p10le",
        ]
    )


def test_cli_hdr2sdr(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
            "--depth-model",
            "Any_S",
            "--colorspace",
            "bt709-tv",
            "--video-codec",
            "libx265",
            "--pix-fmt",
            "yuv420p",
        ]
    )


def test_cli_inpaint(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--method", "forward_inpaint"])


def test_cli_inpaint_vda(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "VDA_S",
            "--method",
            "mlbw_l2_inpaint",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )


def test_cli_export_import(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video,
            "--depth-model",
            "Any_S",
            "--export-disparity",
            "--export-depth-only",
            "--export-depth-fit",
        ]
    )
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_hdr,
            "--depth-model",
            "VDA_S",
            "--export",
            "--ema-normalize",
            "--ema-buffer",
            "10",
            "--scene-detect",
        ]
    )
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image,
            "--depth-model",
            "Any_S",
            "--export",
            "--depth-aa",
            "--export-depth-fit",
        ]
    )

    hdr_yaml = os.path.join(cli_setup.output, "hdr", "iw3_export.yml")
    sd_yaml = os.path.join(cli_setup.output, "iw3_export.yml")
    if os.path.exists(hdr_yaml):
        run([*cli_setup.cli, "-i", hdr_yaml])
    if os.path.exists(sd_yaml):
        run([*cli_setup.cli, "-i", sd_yaml])


def test_cli_vf(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--vf", "scale=-2:320,crop=256:256"])


def test_cli_keyframe(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video, "--depth-model", "Any_S", "--keyframe"])
    out_dir = os.path.join(cli_setup.output, "sd_LRF_Full_SBS")
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    assert len(files) > 0
    with Image.open(os.path.join(out_dir, files[0])) as im:
        assert im.mode == "RGB"


def test_cli_video_rgba_keyframe(cli_setup: CliSetup) -> None:
    run([*cli_setup.cli, "-i", cli_setup.video_rgba, "--depth-model", "Any_S", "--keyframe"])
    out_dir = os.path.join(cli_setup.output, "rgba_LRF_Full_SBS")
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    assert len(files) > 0
    with Image.open(os.path.join(out_dir, files[0])) as im:
        assert im.mode == "RGBA"
        assert np.array(im)[:, :, 3].mean() < 240


def test_cli_video_rgba_vp9_keyframe(cli_setup: CliSetup) -> None:
    if not VU.has_vp9() or cli_setup.video_rgba_vp9 is None:
        pytest.skip("libvpx-vp9 is not supported")
    run([*cli_setup.cli, "-i", cli_setup.video_rgba_vp9, "--depth-model", "Any_S", "--keyframe"])
    out_dir = os.path.join(cli_setup.output, "rgba_vp9_LRF_Full_SBS")
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    assert len(files) > 0
    with Image.open(os.path.join(out_dir, files[0])) as im:
        assert im.mode == "RGBA"
        assert np.array(im)[:, :, 3].mean() < 240


def test_cli_image_rgba(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_rgba.png")
    run([*cli_setup.cli, "-i", cli_setup.image_rgba, "-o", out_file, "--depth-model", "Any_S"])
    with Image.open(out_file) as im:
        assert im.mode == "RGBA"
        arr = np.array(im)
        zero_alpha = arr[:, :, 3] == 0
        assert zero_alpha.any()
        assert not np.all(arr[zero_alpha, :3] == [0, 255, 0])

    out_jpg = os.path.join(cli_setup.output, "out_rgba.jpg")
    run([*cli_setup.cli, "-i", cli_setup.image_rgba, "-o", out_jpg, "--depth-model", "Any_S"])
    with Image.open(out_jpg) as im:
        assert im.mode == "RGB"
        jpg_arr = np.array(im)
        assert jpg_arr[zero_alpha, 1].mean() > 200
        assert jpg_arr[zero_alpha, 0].mean() < 50
        assert jpg_arr[zero_alpha, 2].mean() < 50


def test_cli_image_rgba_rgbd_anaglyph(cli_setup: CliSetup) -> None:
    out_rgbd = os.path.join(cli_setup.output, "out_rgba_rgbd.png")
    run([*cli_setup.cli, "-i", cli_setup.image_rgba, "-o", out_rgbd, "--depth-model", "Any_S", "--rgbd"])
    with Image.open(out_rgbd) as im:
        assert im.mode == "RGBA"
        arr = np.array(im)
        zero_alpha = arr[:, :, 3] == 0
        assert zero_alpha.any()
        assert not np.all(arr[zero_alpha, :3] == [0, 255, 0])

    out_anaglyph = os.path.join(cli_setup.output, "out_rgba_anaglyph.png")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image_rgba,
            "-o",
            out_anaglyph,
            "--depth-model",
            "Any_S",
            "--anaglyph",
            "dubois",
        ]
    )
    with Image.open(out_anaglyph) as im:
        assert im.mode == "RGB"


def test_cli_image_rgba_preprocess(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_rgba_prep.png")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--max-output-height",
            "320",
            "--rotate-left",
        ]
    )
    with Image.open(out_file) as im:
        assert im.mode == "RGBA"
        assert im.height == 320


def test_cli_image_rgba_autocrop(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_rgba_crop.png")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--autocrop",
            "BLACK",
        ]
    )
    with Image.open(out_file) as im:
        assert im.mode == "RGBA"


def test_cli_image_rgba_sod(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_rgba_sod.png")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.image_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--convergence-mode",
            "sod_v1",
        ]
    )
    with Image.open(out_file) as im:
        assert im.mode == "RGBA"


def test_cli_video_rgba_ffv1(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_video_rgba_ffv1.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--video-codec",
            "ffv1",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "ffv1"
        assert vs.pix_fmt == "yuva420p"
        f = next(container.decode(vs))
        rgba = f.to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_video_rgba_vp9(cli_setup: CliSetup) -> None:
    if not VU.has_vp9():
        pytest.skip("libvpx-vp9 is not supported")
    out_file = os.path.join(cli_setup.output, "out_video_rgba_vp9.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--video-codec",
            "libvpx-vp9",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "vp9"
        assert vs.metadata.get("alpha_mode") == "1"


def test_cli_video_webm(cli_setup: CliSetup) -> None:
    if not VU.has_vp9():
        pytest.skip("libvpx-vp9 is not supported")
    out_file = os.path.join(cli_setup.output, "out_video.webm")
    run([*cli_setup.cli, "-i", cli_setup.video, "-o", out_file, "--depth-model", "Any_S"])
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "vp9"
        assert len(container.streams.audio) > 0
        audio_stream = container.streams.audio[0]
        assert audio_stream.codec_context.name in {"opus", "libopus"}
        assert audio_stream.rate == 48000


def test_cli_video_rgba_webm(cli_setup: CliSetup) -> None:
    if not VU.has_vp9():
        pytest.skip("libvpx-vp9 is not supported")
    out_file = os.path.join(cli_setup.output, "out_video_rgba.webm")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "vp9"
        assert vs.metadata.get("alpha_mode") == "1"
        decoder = VU.VideoDecoder(vs)
        frames = []
        for packet in container.demux(vs):
            for frame in decoder.decode(packet):
                frames.append(frame)
                break
            if frames:
                break
        assert len(frames) > 0
        rgba = frames[0].to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_video_rgba_input_vp9(cli_setup: CliSetup) -> None:
    if not VU.has_vp9() or cli_setup.video_rgba_vp9 is None:
        pytest.skip("libvpx-vp9 is not supported")
    out_file = os.path.join(cli_setup.output, "out_video_rgba_from_vp9.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba_vp9,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--video-codec",
            "ffv1",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "ffv1"
        assert vs.pix_fmt == "yuva420p"
        f = next(container.decode(vs))
        rgba = f.to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_video_rgba_low_vram(cli_setup: CliSetup) -> None:
    out_file = os.path.join(cli_setup.output, "out_video_rgba_low_vram.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba,
            "-o",
            out_file,
            "--depth-model",
            "Any_S",
            "--low-vram",
            "--ema-normalize",
            "--video-codec",
            "ffv1",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "ffv1"
        assert vs.pix_fmt == "yuva420p"
        f = next(container.decode(vs))
        rgba = f.to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_video_rgba_export_import(cli_setup: CliSetup) -> None:
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba,
            "-o",
            cli_setup.output,
            "--depth-model",
            "Any_S",
            "--export",
        ]
    )
    export_dir = os.path.join(cli_setup.output, "rgba")
    rgb_dir = os.path.join(export_dir, "rgb")
    depth_dir = os.path.join(export_dir, "depth")
    yaml_file = os.path.join(export_dir, "iw3_export.yml")
    assert os.path.exists(yaml_file)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])
    assert len(rgb_files) > 0
    assert len(rgb_files) == len(depth_files)

    with Image.open(os.path.join(rgb_dir, rgb_files[0])) as im:
        assert im.mode == "RGBA"
        assert np.array(im)[:, :, 3].mean() < 240

    out_file = os.path.join(cli_setup.output, "out_video_rgba_imported.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            yaml_file,
            "-o",
            out_file,
            "--video-codec",
            "ffv1",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "ffv1"
        assert vs.pix_fmt == "yuva420p"
        f = next(container.decode(vs))
        rgba = f.to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_video_rgba_vp9_export_import(cli_setup: CliSetup) -> None:
    if not VU.has_vp9() or cli_setup.video_rgba_vp9 is None:
        pytest.skip("libvpx-vp9 is not supported")
    run(
        [
            *cli_setup.cli,
            "-i",
            cli_setup.video_rgba_vp9,
            "-o",
            cli_setup.output,
            "--depth-model",
            "Any_S",
            "--export",
        ]
    )
    export_dir = os.path.join(cli_setup.output, "rgba_vp9")
    rgb_dir = os.path.join(export_dir, "rgb")
    depth_dir = os.path.join(export_dir, "depth")
    yaml_file = os.path.join(export_dir, "iw3_export.yml")
    assert os.path.exists(yaml_file)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])
    assert len(rgb_files) > 0
    assert len(rgb_files) == len(depth_files)

    with Image.open(os.path.join(rgb_dir, rgb_files[0])) as im:
        assert im.mode == "RGBA"
        assert np.array(im)[:, :, 3].mean() < 240

    out_file = os.path.join(cli_setup.output, "out_video_rgba_vp9_imported.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            yaml_file,
            "-o",
            out_file,
            "--video-codec",
            "libvpx-vp9",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.codec_context.name == "vp9"
        assert vs.metadata.get("alpha_mode") == "1"
        decoder = VU.VideoDecoder(vs)
        frames = []
        for packet in container.demux(vs):
            for frame in decoder.decode(packet):
                frames.append(frame)
                break
            if frames:
                break
        assert len(frames) > 0
        rgba = frames[0].to_ndarray(format="rgba")
        assert rgba[:, :, 3].mean() < 240


def test_cli_image_rgba_inpaint(cli_setup: CliSetup) -> None:
    for method in ("forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"):
        out_file = os.path.join(cli_setup.output, f"out_rgba_{method}.png")
        run([*cli_setup.cli, "-i", cli_setup.image_rgba, "-o", out_file, "--depth-model", "Any_S", "--method", method])
        with Image.open(out_file) as im:
            assert im.mode == "RGBA"
            arr = np.array(im)
            assert (arr[:, :, 3] == 0).any()


def test_cli_video_rgba_inpaint(cli_setup: CliSetup) -> None:
    for method in ("forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"):
        out_file = os.path.join(cli_setup.output, f"out_video_rgba_{method}.mkv")
        run(
            [
                *cli_setup.cli,
                "-i",
                cli_setup.video_rgba,
                "-o",
                out_file,
                "--depth-model",
                "Any_S",
                "--method",
                method,
                "--video-codec",
                "ffv1",
                "--pix-fmt",
                "yuva420p",
            ]
        )
        with av.open(out_file) as container:
            vs = container.streams.video[0]
            assert vs.pix_fmt == "yuva420p"
            frames = list(container.decode(vs))
            assert len(frames) > 0
            for frame in frames:
                rgba = frame.to_ndarray(format="rgba")
                assert (rgba[:, :, 3] == 0).any()


def test_cli_video_rgba_export_import_inpaint(cli_setup: CliSetup) -> None:
    export_dir = os.path.join(cli_setup.output, "rgba")
    yaml_file = os.path.join(export_dir, "iw3_export.yml")
    if not os.path.exists(yaml_file):
        test_cli_video_rgba_export_import(cli_setup)

    out_file = os.path.join(cli_setup.output, "out_video_rgba_imported_inpaint.mkv")
    run(
        [
            *cli_setup.cli,
            "-i",
            yaml_file,
            "-o",
            out_file,
            "--method",
            "forward_inpaint",
            "--video-codec",
            "ffv1",
            "--pix-fmt",
            "yuva420p",
        ]
    )
    with av.open(out_file) as container:
        vs = container.streams.video[0]
        assert vs.pix_fmt == "yuva420p"
        frames = list(container.decode(vs))
        assert len(frames) > 0
        for f in frames:
            rgba = f.to_ndarray(format="rgba")
            assert (rgba[:, :, 3] == 0).any()


def test_video_inpaint_alpha_memory_corruption() -> None:
    import torch

    from iw3.forward_inpaint import ForwardInpaintVideo

    vid_model = ForwardInpaintVideo(device_id=-1, pre_padding=1, post_padding=1)
    vid_model.model_seq = 4

    outputs = []
    for i in range(5):
        x = torch.full((1, 3, 16, 16), float(i))
        depth = torch.ones((1, 1, 16, 16)) * 0.5
        alpha = torch.full((1, 1, 16, 16), float(i + 1) / 10.0)
        ret = vid_model.infer(x, depth, divergence=0.0, convergence=0.5, synthetic_view="both", alpha=alpha)
        if ret[0] is not None:
            assert len(ret) == 4
            assert ret[0] is not None and ret[2] is not None
            outputs.append((ret[0], ret[2]))  # left_eye, left_alpha

    ret = vid_model.flush()
    if ret[0] is not None:
        assert len(ret) == 4
        assert ret[0] is not None and ret[2] is not None
        outputs.append((ret[0], ret[2]))

    # Flatten all emitted frames
    all_left_eye = torch.cat([o[0] for o in outputs], dim=0)
    all_left_alpha = torch.cat([o[1] for o in outputs], dim=0)
    assert all_left_eye.shape[0] == 5
    assert all_left_alpha.shape[0] == 5

    # Check each frame's alpha value is not corrupted by subsequent frame additions
    for i in range(5):
        expected_alpha = float(i + 1) / 10.0
        assert torch.allclose(all_left_alpha[i], torch.tensor(expected_alpha), atol=1e-4)


def test_inpaint_warp_alpha_no_foreground_holes() -> None:
    import torch

    from iw3.forward_inpaint import ForwardInpaintImage
    from iw3.mlbw_inpaint import MLBWInpaintImage
    from iw3.monobw_inpaint import MonoBWInpaintImage

    h, w = 256, 256
    alpha = torch.zeros(1, 1, h, w)
    alpha[:, :, 40:216, 40:216] = 1.0  # foreground object (clock face)

    depth = torch.ones(1, 1, h, w) * 0.1
    depth[:, :, 40:216, 40:216] = 0.5  # object base depth
    depth[:, :, 60:196, 120:136] = 0.9  # depth step inside object (clock hand)

    models = [
        ForwardInpaintImage(device_id=-1),
        MLBWInpaintImage(device_id=-1),
        MonoBWInpaintImage(device_id=-1),
    ]

    for model in models:
        left_alpha, right_alpha = model.warp_alpha(alpha, depth, divergence=5.0, convergence=0.5, synthetic_view="both")
        # The region [80:180, 80:160] is completely inside the foreground face (Alpha == 1)
        # Even across depth steps, holes (Alpha == 0) must not appear inside the foreground
        assert (left_alpha[:, :, 80:180, 80:160] == 0).sum().item() == 0
        assert (right_alpha[:, :, 80:180, 80:160] == 0).sum().item() == 0
