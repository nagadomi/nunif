import pytest
import torch
import torch.nn as nn
from PIL import Image

from iw3.base_depth_model import BaseDepthModel
from iw3.depth_models.null_depth_model import NullDepthModel
from nunif.models.data_parallel import DeviceSwitchInference


class _RecordingModule(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.seen_shapes: list[tuple[int, ...]] = []
        self.seen_is_inference: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(tuple(x.shape))
        self.seen_is_inference.append(x.is_inference())
        return self.inner(x)


class _DummyStreamingModel(NullDepthModel):
    def is_streaming(self) -> bool:
        return True


def _create_null_model(resolution: int = 32) -> tuple[NullDepthModel, _RecordingModule]:
    model = NullDepthModel("NULL")
    model.device = torch.device("cpu")
    recorder = _RecordingModule(model.load_model("NULL", resolution=resolution))
    model.model = recorder
    return model, recorder


def test_force_batch_size_3d_and_pil() -> None:
    model, recorder = _create_null_model()
    x_3d = torch.rand(3, 32, 32)
    ref = model.infer(x_3d)
    assert ref.shape == (1, 32, 32)

    with model.force_batch_size(4):
        out = model.infer(x_3d)
    assert out.shape == (1, 32, 32)
    assert recorder.seen_shapes[-1] == (4, 3, 32, 32)
    assert torch.allclose(out, ref)

    im = Image.new("RGB", (32, 32), color=(128, 64, 32))
    ref_pil = model.infer(im)
    with model.force_batch_size(4):
        out_pil = model.infer(im)
    assert out_pil.shape == (1, 32, 32)
    assert recorder.seen_shapes[-1] == (4, 3, 32, 32)
    assert torch.allclose(out_pil, ref_pil)


def test_force_batch_size_4d_and_alpha() -> None:
    model, recorder = _create_null_model()
    x_4d = torch.rand(2, 3, 32, 32)
    alpha_4d = torch.zeros(2, 1, 32, 32)
    alpha_4d[:, :, 8:24, 8:24] = 1.0

    ref = model.infer(x_4d, alpha=alpha_4d)
    assert ref.shape == (2, 1, 32, 32)

    with model.force_batch_size(4):
        out = model.infer(x_4d, alpha=alpha_4d)
    assert out.shape == (2, 1, 32, 32)
    assert recorder.seen_shapes[-1] == (4, 3, 32, 32)
    assert torch.allclose(out, ref)


def test_force_batch_size_preserves_inference_mode() -> None:
    model, recorder = _create_null_model()
    x_non_inf = torch.rand(2, 3, 32, 32)
    assert not x_non_inf.is_inference()

    with model.force_batch_size(4):
        model.infer(x_non_inf)
        assert recorder.seen_is_inference[-1] is False

        with torch.inference_mode():
            x_inf = torch.rand(2, 3, 32, 32)
            assert x_inf.is_inference()
            model.infer(x_inf)
            assert recorder.seen_is_inference[-1] is True


def test_force_batch_size_skipped_for_streaming() -> None:
    model = _DummyStreamingModel("NULL")
    model.device = torch.device("cpu")
    recorder = _RecordingModule(model.load_model("NULL", resolution=32))
    model.model = recorder

    x_4d = torch.rand(2, 3, 32, 32)
    with model.force_batch_size(4):
        out = model.infer(x_4d)
    assert out.shape == (2, 1, 32, 32)
    assert recorder.seen_shapes[-1] == (2, 3, 32, 32)


def test_compile_context_force_batch_size() -> None:
    model, _ = _create_null_model()
    assert model._force_batch_size is None

    with model.compile_context(enabled=False, force_batch_size=4):
        assert model._force_batch_size is None

    orig_compile = model.compile
    orig_clear = model.clear_compiled_model
    model.compile = lambda: None  # type: ignore[method-assign]
    model.clear_compiled_model = lambda: None  # type: ignore[method-assign]
    try:
        with model.compile_context(enabled=True, force_batch_size=4):
            assert model._force_batch_size == 4
        assert model._force_batch_size is None

        model.model = DeviceSwitchInference.__new__(DeviceSwitchInference)
        with model.compile_context(enabled=True, force_batch_size=4):
            assert model._force_batch_size is None
    finally:
        model.compile = orig_compile  # type: ignore[method-assign]
        model.clear_compiled_model = orig_clear  # type: ignore[method-assign]


def test_restore_infer_output_tuple() -> None:
    depth = torch.zeros(4, 1, 8, 8)
    mask = torch.ones(4, 1, 8, 8)
    d_out, m_out = BaseDepthModel._restore_infer_output((depth, mask), is_batch=True, orig_batch_size=2)
    assert d_out.shape == (2, 1, 8, 8)
    assert m_out.shape == (2, 1, 8, 8)

    d_3d, m_3d = BaseDepthModel._restore_infer_output((depth, mask), is_batch=False, orig_batch_size=1)
    assert d_3d.shape == (1, 8, 8)
    assert m_3d.shape == (1, 8, 8)


def test_force_batch_size_assertions() -> None:
    model, _ = _create_null_model()

    with pytest.raises(AssertionError):
        model.force_batch_size(0)

    with model.force_batch_size(4):
        with pytest.raises(AssertionError):
            model.infer(torch.rand(5, 3, 32, 32))

        with pytest.raises(AssertionError):
            model.infer(torch.rand(0, 3, 32, 32))

        with pytest.raises(AssertionError):
            model.infer(torch.rand(2, 3, 32, 32), alpha=torch.ones(1, 32, 32))

        with pytest.raises(AssertionError):
            model._prepare_infer_input(torch.rand(2, 3, 32, 32), alpha=[torch.ones(1, 32, 32)])


def test_force_batch_size_nested_and_exception() -> None:
    model, _ = _create_null_model()
    assert model._force_batch_size is None

    with model.force_batch_size(4):
        assert model._force_batch_size == 4
        with model.force_batch_size(2):
            assert model._force_batch_size == 2
        assert model._force_batch_size == 4

        with pytest.raises(RuntimeError):
            with model.force_batch_size(8):
                assert model._force_batch_size == 8
                raise RuntimeError("boom")
        assert model._force_batch_size == 4

    assert model._force_batch_size is None


def test_force_batch_size_tta_and_list_alpha() -> None:
    model, recorder = _create_null_model()
    x_4d = torch.rand(2, 3, 32, 32)
    alpha_list: list[torch.Tensor | None] = [torch.ones(1, 32, 32), None]

    ref = model.infer(x_4d, tta=True)
    with model.force_batch_size(4):
        out = model.infer(x_4d, tta=True)
    assert out.shape == (2, 1, 32, 32)
    assert recorder.seen_shapes[-1] == (4, 3, 32, 32)
    assert torch.allclose(out, ref)

    x_pad, alpha_pad, is_batch, orig_b = model._prepare_infer_input(x_4d, alpha=alpha_list)
    assert is_batch is True
    assert orig_b == 2
    assert x_pad.shape[0] == 2
    assert alpha_pad == alpha_list

    with model.force_batch_size(4):
        x_pad, alpha_pad, is_batch, orig_b = model._prepare_infer_input(x_4d, alpha=alpha_list)
    assert is_batch is True
    assert orig_b == 2
    assert x_pad.shape[0] == 4
    assert isinstance(alpha_pad, list) and len(alpha_pad) == 4
    assert alpha_pad[2] is None and alpha_pad[3] is None
