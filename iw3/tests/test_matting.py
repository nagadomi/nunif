import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from iw3.base_depth_model import BaseDepthModel
from iw3.depth_model_factory import is_matting
from iw3.utils import (
    GREEN_FILL_COLOR,
    bind_batch_frame_callback,
    bind_export_single_frame_callback,
    bind_single_frame_callback,
    create_parser,
    process_image,
    set_state_args,
)


class DummyMattingModel(BaseDepthModel):
    def __init__(self, model_type: str = "DummyMatting"):
        super().__init__(model_type)
        self.device = torch.device("cpu")

    @classmethod
    def is_matting(cls, model_type: str | None = None) -> bool:
        return True

    @classmethod
    def is_desktop_supported(cls) -> bool:
        return False

    @classmethod
    def get_name(cls) -> str:
        return "DummyMatting"

    @classmethod
    def supported(cls, model_type: str) -> bool:
        return model_type == "DummyMatting"

    @classmethod
    def get_model_types(cls) -> list[str]:
        return ["DummyMatting"]

    @classmethod
    def has_checkpoint_file(cls, model_type: str) -> bool:
        return True

    @classmethod
    def get_model_path(cls, model_type: str) -> str:
        return ""

    @classmethod
    def multi_gpu_supported(cls, model_type: str) -> bool:
        return False

    @classmethod
    def force_update(cls) -> None:
        pass

    def load_model(self, *args, **kwargs):
        return torch.nn.Identity()

    def is_metric(self) -> bool:
        return False

    def infer(
        self,
        x: torch.Tensor | Image.Image,
        tta: bool = False,
        low_vram: bool = False,
        enable_amp: bool = True,
        edge_dilation: int | tuple[int, int] = 0,
        depth_aa: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(x):
            from torchvision.transforms import functional as TF

            x = TF.to_tensor(x)
        assert torch.is_tensor(x)
        # depth: 0.5 in foreground, 0.1 in background
        # mask: 1.0 in center region, 0.0 in border
        h, w = x.shape[-2:]
        if x.ndim == 4:
            b = x.shape[0]
            depth = torch.full((b, 1, h, w), 0.1, dtype=torch.float32)
            mask = torch.zeros((b, 1, h, w), dtype=torch.float32)
            depth[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.5
            mask[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        else:
            depth = torch.full((1, h, w), 0.1, dtype=torch.float32)
            mask = torch.zeros((1, h, w), dtype=torch.float32)
            depth[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.5
            mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        return depth.to(x.device), mask.to(x.device)


def _create_test_args(**kwargs) -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args(["-i", "dummy.png", "-o", "dummy_out.png", "--mapper", "none"])
    for k, v in kwargs.items():
        setattr(args, k, v)
    set_state_args(args)
    return args


def test_is_matting_interface() -> None:
    assert BaseDepthModel.is_matting("DummyMatting") is True
    assert is_matting("DummyMatting") is True
    assert BaseDepthModel.is_matting("Any_S") is False
    assert is_matting("Any_S") is False
    assert BaseDepthModel.is_matting(None) is False

    dummy = DummyMattingModel()
    assert dummy.is_matting() is True


def test_matting_process_image_rgba() -> None:
    args = _create_test_args(format="png", method="forward_fill")
    depth_model = DummyMattingModel()

    # 1. RGB input
    rgb = torch.ones((3, 32, 32), dtype=torch.float32)
    out = process_image(rgb, args, depth_model, side_model=None)
    # Output must have alpha channel (4 channels: SBS RGBA)
    assert out.shape[0] == 4
    # Center region should have alpha == 1.0, border should have alpha == 0.0
    alpha = out[3]
    assert (alpha[:, 8:24] == 1.0).any()
    assert (alpha[:, :4] == 0.0).all()

    # 2. Existing Alpha input
    input_alpha = torch.zeros((1, 32, 32), dtype=torch.float32)
    input_alpha[:, 12:20, 12:20] = 1.0
    out_with_alpha = process_image(rgb, args, depth_model, side_model=None, alpha=input_alpha)
    assert out_with_alpha.shape[0] == 4
    # Alpha should be input_alpha * model_mask (intersection)
    res_alpha = out_with_alpha[3]
    assert (res_alpha[:, 12:20] == 1.0).any()
    assert (res_alpha[:, :8] == 0.0).all()


def test_matting_process_image_green_fill() -> None:
    # When format is JPEG (no alpha support), transparent region must be filled with green
    args = _create_test_args(format="jpeg", method="forward_fill")
    depth_model = DummyMattingModel()

    rgb = torch.ones((3, 32, 32), dtype=torch.float32)
    out = process_image(rgb, args, depth_model, side_model=None)
    # Output is 3ch RGB with green background
    assert out.shape[0] == 3
    # Border (where alpha was 0) should be green (0.0, 1.0, 0.0)
    for c in range(3):
        assert torch.allclose(out[c, :4, :4], torch.tensor(GREEN_FILL_COLOR[c]), atol=1e-3)


def test_matting_single_frame_callback() -> None:
    args = _create_test_args(video_format="webm", video_codec="libvpx-vp9", pix_fmt="yuva420p", method="forward_fill")
    depth_model = DummyMattingModel()

    cb = bind_single_frame_callback(depth_model, side_model=None, segment_pts=set(), args=args)

    frames = []
    for i in range(3):
        f = av.VideoFrame.from_ndarray(np.ones((32, 32, 3), dtype=np.uint8) * 200, format="rgb24")
        f.pts = i
        for out in cb(f):
            frames.append(out)

    for out in cb(None):  # flush
        frames.append(out)

    assert len(frames) == 3
    for frame in frames:
        # yuva420p output must have alpha channel (4 channels)
        assert frame.shape[0] == 4


def test_matting_batch_frame_callback() -> None:
    args = _create_test_args(
        video_format="webm",
        video_codec="libvpx-vp9",
        pix_fmt="yuva420p",
        batch_size=2,
        cuda_stream=False,
        method="forward_fill",
    )
    depth_model = DummyMattingModel()

    infer_fn, preprocess_fn = bind_batch_frame_callback(depth_model, side_model=None, segment_pts=set(), args=args)

    frames = []
    x = torch.ones((2, 3, 32, 32), dtype=torch.float32)
    prep_args = preprocess_fn(x, [0, 1], False)
    frames.extend(infer_fn(prep_args))

    flush_args = preprocess_fn(None, None, True)
    frames.extend(list(infer_fn(flush_args)))

    assert len(frames) == 2
    for frame in frames:
        assert frame.shape[0] == 4


def test_matting_export_callback(tmp_path: Path) -> None:
    rgb_dir = str(tmp_path / "rgb")
    depth_dir = str(tmp_path / "depth")
    import os

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    args = _create_test_args()
    depth_model = DummyMattingModel()

    with PoolExecutor(max_workers=2) as pool:
        cb = bind_export_single_frame_callback(depth_model, set(), rgb_dir, depth_dir, pool, args)
        f = av.VideoFrame.from_ndarray(np.ones((32, 32, 3), dtype=np.uint8) * 200, format="rgb24")
        f.pts = 0
        cb(f)
        cb(None)  # flush

    # Verify rgb file was saved as RGBA PNG
    rgb_path = Path(rgb_dir) / "00000000.png"
    assert rgb_path.exists()
    im = Image.open(rgb_path)
    assert im.mode == "RGBA"
