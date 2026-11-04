import os

import av
import pytest

from nunif.utils.video import has_vp9
from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data

MODEL_DIR = "waifu2x/pretrained_models/upconv_7/art"


@pytest.fixture(scope="module")
def cli_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_dir = tmp_path_factory.mktemp("cli")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    run(["python", "-m", "waifu2x.download_models"])
    return {
        "image": os.path.join(smoke_dir, "sd.png"),
        "video": os.path.join(smoke_dir, "sd.mkv"),
        "video_rgba": os.path.join(smoke_dir, "rgba.mkv"),
        "dir": smoke_dir,
        "output": output_dir,
    }


def test_cli_image(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["image"],
            "--noise-level",
            "0",
            "-m",
            "noise_scale",
        ]
    )


def test_cli_video(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["video"],
            "--noise-level",
            "1",
            "-m",
            "noise",
        ]
    )


def test_cli_dir(cli_setup: dict[str, str]) -> None:
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            cli_setup["output"],
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["dir"],
            "-m",
            "scale",
        ]
    )


def test_cli_video_rgba_ffv1(cli_setup: dict[str, str]) -> None:
    out_file = os.path.join(cli_setup["output"], "out_video_rgba_ffv1.mkv")
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            out_file,
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["video_rgba"],
            "-m",
            "scale",
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


def test_cli_video_rgba_vp9(cli_setup: dict[str, str]) -> None:
    if not has_vp9():
        pytest.skip("libvpx-vp9 is not supported")
    out_file = os.path.join(cli_setup["output"], "out_video_rgba_vp9.mkv")
    run(
        [
            "python",
            "-m",
            "waifu2x.cli",
            "-y",
            "-o",
            out_file,
            "--model-dir",
            MODEL_DIR,
            "-i",
            cli_setup["video_rgba"],
            "-m",
            "scale",
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
