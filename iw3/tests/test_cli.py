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
