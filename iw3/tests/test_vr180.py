import os
from typing import NamedTuple

import av
import numpy as np
import pytest
import torch

from iw3.utils import create_parser, postprocess_image, set_state_args
from tests.helpers.runner import run
from tests.helpers.smoke_data import ensure_smoke_data


class VR180Setup(NamedTuple):
    video: str
    output_dir: str
    cli: list[str]


@pytest.fixture(scope="module")
def vr180_setup(tmp_path_factory: pytest.TempPathFactory) -> VR180Setup:
    tmp_dir = tmp_path_factory.mktemp("vr180")
    smoke_dir = str(tmp_dir / "smoke")
    output_dir = str(tmp_dir / "output")
    ensure_smoke_data(smoke_dir)
    return VR180Setup(
        video=os.path.join(smoke_dir, "sd.mkv"),
        output_dir=output_dir,
        cli=["python", "-m", "iw3.cli", "-y", "-o", output_dir],
    )


def test_vr180_postprocess_batch_parity() -> None:
    torch.manual_seed(42)
    b, c, h, w = 4, 3, 256, 256
    left_batch = torch.rand(b, c, h, w)
    right_batch = torch.rand(b, c, h, w)
    left_alpha = torch.rand(b, 1, h, w)
    right_alpha = torch.rand(b, 1, h, w)

    test_configs = [
        ["--vr180"],
        ["--vr180", "--half-sbs"],
        ["--vr180", "--tb"],
        ["--vr180", "--half-tb"],
        ["--vr180", "--max-output-width", "512", "--max-output-height", "256"],
    ]

    for cfg in test_configs:
        args = set_state_args(create_parser().parse_args(["-i", "dummy", "-o", "dummy", *cfg]))

        # Test RGB batch parity
        batch_out = postprocess_image(left_batch, right_batch, args)
        single_out = [postprocess_image(left_batch[i], right_batch[i], args) for i in range(b)]
        assert len(batch_out) == b
        for i in range(b):
            assert torch.equal(batch_out[i], single_out[i])

        # Test RGBA (with alpha channels) batch parity
        batch_out_rgba = postprocess_image(
            left_batch,
            right_batch,
            args,
            left_alpha=left_alpha,
            right_alpha=right_alpha,
        )
        single_out_rgba = [
            postprocess_image(
                left_batch[i],
                right_batch[i],
                args,
                left_alpha=left_alpha[i],
                right_alpha=right_alpha[i],
            )
            for i in range(b)
        ]
        assert len(batch_out_rgba) == b
        for i in range(b):
            assert torch.equal(batch_out_rgba[i], single_out_rgba[i])


def test_vr180_cli_batch_parity(vr180_setup: VR180Setup) -> None:
    out_b1 = os.path.join(vr180_setup.output_dir, "vr180_b1.mkv")
    out_b4 = os.path.join(vr180_setup.output_dir, "vr180_b4.mkv")

    common_opts = [
        "-i",
        vr180_setup.video,
        "--depth-model",
        "Any_S",
        "--vr180",
        "--video-codec",
        "ffv1",
        "--max-output-width",
        "640",
    ]

    run([*vr180_setup.cli, *common_opts, "-o", out_b1, "--batch-size", "1"])
    run([*vr180_setup.cli, *common_opts, "-o", out_b4, "--batch-size", "4"])

    with av.open(out_b1) as c1, av.open(out_b4) as c4:
        frames_b1 = [f.to_ndarray() for f in c1.decode(video=0)]
        frames_b4 = [f.to_ndarray() for f in c4.decode(video=0)]

    assert len(frames_b1) == len(frames_b4)
    assert len(frames_b1) > 0

    for idx, (f1, f4) in enumerate(zip(frames_b1, frames_b4)):
        assert f1.shape == f4.shape
        diff = np.abs(f1.astype(int) - f4.astype(int))
        assert diff.max() <= 1
        mse = np.mean((f1.astype(float) - f4.astype(float)) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
        assert psnr >= 70.0
