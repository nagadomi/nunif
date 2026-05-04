import contextlib
from abc import ABC, abstractmethod
from typing import Callable, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.device import autocast, create_device
from nunif.models.utils import compile_function

from .inpaint_utils import CompileContext
from .models import LightInpaintV1, LightVideoInpaintV1


class FrameQueue:
    def __init__(self, synthetic_view, seq, height, width, dtype, device, mask_height=None, mask_width=None):
        if mask_width is None:
            mask_width = width
        if mask_height is None:
            mask_height = height

        self.left_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        self.right_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        if synthetic_view == "both":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
        elif synthetic_view == "right":
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.left_mask = None
        elif synthetic_view == "left":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = None

        self.synthetic_view = synthetic_view
        self.index = 0
        self.max_index = seq

    def full(self):
        return self.index == self.max_index

    def empty(self):
        return self.index == 0

    def add(self, left_eye, right_eye, left_mask=None, right_mask=None):
        self.left_eye[self.index] = left_eye
        self.right_eye[self.index] = right_eye
        if left_mask is not None:
            self.left_mask[self.index] = left_mask
        if right_mask is not None:
            self.right_mask[self.index] = right_mask

        self.index += 1

    def fill(self):
        if self.full():
            return 0

        pad = 0
        i = self.index - 1
        if self.synthetic_view == "both":
            frame = dict(
                left_eye=self.left_eye[i].clone(),
                right_eye=self.right_eye[i].clone(),
                left_mask=self.left_mask[i].clone(),
                right_mask=self.right_mask[i].clone(),
            )
        elif self.synthetic_view == "right":
            frame = dict(
                left_eye=self.left_eye[i].clone(),
                right_eye=self.right_eye[i].clone(),
                right_mask=self.right_mask[i].clone(),
            )
        elif self.synthetic_view == "left":
            frame = dict(
                left_eye=self.left_eye[i].clone(),
                right_eye=self.right_eye[i].clone(),
                left_mask=self.left_mask[i].clone(),
            )
        while not self.full():
            pad += 1
            self.add(**frame)

        return pad

    def remove(self, n):
        keep_count = self.index - n
        for i in range(keep_count):
            self.left_eye[i] = self.left_eye[i + n]
            self.right_eye[i] = self.right_eye[i + n]
            if self.right_mask is not None:
                self.right_mask[i] = self.right_mask[i + n]
            if self.left_mask is not None:
                self.left_mask[i] = self.left_mask[i + n]
        self.index -= n
        assert self.index >= 0

    def get(self):
        if self.synthetic_view == "both":
            return self.left_eye, self.right_eye, self.left_mask, self.right_mask
        elif self.synthetic_view == "left":
            return self.left_eye, self.right_eye, self.left_mask
        elif self.synthetic_view == "right":
            return self.left_eye, self.right_eye, self.right_mask

    def clear(self):
        self.index = 0


class InpaintComponent(nn.Module, ABC):
    # interface

    def reset(self) -> None:
        pass

    def flush(
        self, inner_dilation: int = 0, outer_dilation: int = 0, enable_amp: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def compile(self) -> None:
        pass

    def clear_compiled_model(self) -> None:
        pass

    @abstractmethod
    def infer(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str = "both",
        max_width: int | None = None,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass

    # utils

    @staticmethod
    def _resize(x: torch.Tensor, max_width: int | None) -> torch.Tensor:
        if max_width is not None and x.shape[-1] > max_width:
            if max_width % 2 != 0:
                max_width += 1
            new_w = max_width
            new_h = int((max_width / x.shape[-1]) * x.shape[-2])
            if new_h % 2 != 0:
                new_h += 1
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)
        return x


class BaseImageInpaint(InpaintComponent):
    model: LightInpaintV1 | None

    def __init__(self, model: LightInpaintV1, device_id: int = -1):
        super().__init__()
        self.model = model
        self.device = create_device(device_id)
        self.eval()

    def train(self, mode: bool = True) -> Self:
        return super().train(mode=False)

    @abstractmethod
    def apply_warp(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pass

    @abstractmethod
    def preprocess_mask(
        self,
        mask: torch.Tensor,
        target_size: tuple[int, int] | torch.Size,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        pass

    def _inpaint_single(
        self,
        eye: torch.Tensor,
        mask: torch.Tensor,
        is_left: bool,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        assert self.model is not None
        if is_left:
            eye, mask = eye.flip(-1), mask.flip(-1)

        mask = self.preprocess_mask(
            mask,
            target_size=eye.shape[-2:],
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=base_width,
        )
        eye = self.model.infer(eye, mask)

        if is_left:
            eye = eye.flip(-1)
        return eye

    def _inpaint(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        left_mask: torch.Tensor | None,
        right_mask: torch.Tensor | None,
        synthetic_view: str,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if synthetic_view == "both":
            assert left_mask is not None
            assert right_mask is not None
            left_eye = self._inpaint_single(
                left_eye,
                left_mask,
                is_left=True,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
            right_eye = self._inpaint_single(
                right_eye,
                right_mask,
                is_left=False,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        elif synthetic_view == "right":
            assert right_mask is not None
            right_eye = self._inpaint_single(
                right_eye,
                right_mask,
                is_left=False,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        elif synthetic_view == "left":
            assert left_mask is not None
            left_eye = self._inpaint_single(
                left_eye,
                left_mask,
                is_left=True,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        return left_eye, right_eye

    def infer(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str = "both",
        max_width: int | None = None,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = self._resize(x, max_width)
        return self(
            x,
            depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
        )

    def forward(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str = "both",
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        left_eye, right_eye, left_mask, right_mask = self.apply_warp(
            x,
            depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
        )
        return self._inpaint(
            left_eye,
            right_eye,
            left_mask,
            right_mask,
            synthetic_view,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=depth.shape[-1],
        )


class BaseVideoInpaint(InpaintComponent):
    model: LightVideoInpaintV1 | None
    _model_infer_backup: Callable | None
    frame_queue: FrameQueue | None
    synthetic_view: str | None
    base_width: int | None

    def __init__(self, model: LightVideoInpaintV1, pre_padding: int = 3, post_padding: int = 3, device_id: int = -1):
        super().__init__()
        self.model = model
        self.model_seq = 12
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.frame_queue = None
        self.synthetic_view = None
        self.base_width = None
        self.device = create_device(device_id)
        self.eval()

    def train(self, mode: bool = True) -> Self:
        return super().train(mode=False)

    def reset(self) -> None:
        self.frame_queue = None

    def create_frame_queue(self, x, depth, synthetic_view) -> FrameQueue:
        return FrameQueue(
            synthetic_view=synthetic_view,
            seq=self.model_seq,
            height=x.shape[-2],
            width=x.shape[-1],
            mask_height=depth.shape[-2],
            mask_width=depth.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

    def compile(self) -> None:
        assert self.model is not None
        self._model_infer_backup = self.model.infer
        self.model.infer = compile_function(self.model.infer, device=self.model.get_device())  # type: ignore[assignment]

    def clear_compiled_model(self) -> None:
        assert self.model is not None
        if self._model_infer_backup is not None:
            self.model.infer = self._model_infer_backup  # type: ignore[assignment]
        self._model_infer_backup = None

    @abstractmethod
    def apply_warp(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pass

    @abstractmethod
    def preprocess_mask(
        self,
        mask: torch.Tensor,
        target_size: tuple[int, int] | torch.Size,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        pass

    def _inpaint_single(
        self,
        eye: torch.Tensor,
        mask: torch.Tensor,
        is_left: bool,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        assert self.model is not None
        if is_left:
            eye, mask = eye.flip(-1), mask.flip(-1)

        mask = self.preprocess_mask(
            mask,
            target_size=eye.shape[-2:],
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=base_width,
        )
        eye = self.model.infer(eye, mask)

        if is_left:
            eye = eye.flip(-1)
        return eye

    def _inpaint(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        left_mask: torch.Tensor | None,
        right_mask: torch.Tensor | None,
        synthetic_view: str,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if synthetic_view == "both":
            assert left_mask is not None
            assert right_mask is not None
            left_eye = self._inpaint_single(
                left_eye,
                left_mask,
                is_left=True,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
            right_eye = self._inpaint_single(
                right_eye,
                right_mask,
                is_left=False,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        elif synthetic_view == "right":
            assert right_mask is not None
            right_eye = self._inpaint_single(
                right_eye,
                right_mask,
                is_left=False,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        elif synthetic_view == "left":
            assert left_mask is not None
            left_eye = self._inpaint_single(
                left_eye,
                left_mask,
                is_left=True,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )
        return left_eye, right_eye

    def forward(
        self, flush: bool = False, inner_dilation: int = 0, outer_dilation: int = 0, base_width: int | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.frame_queue is not None
        assert self.synthetic_view is not None
        if self.frame_queue.full():
            left_eye: torch.Tensor
            right_eye: torch.Tensor
            left_mask: torch.Tensor | None = None
            right_mask: torch.Tensor | None = None

            if self.synthetic_view == "both":
                left_eye, right_eye, left_mask, right_mask = self.frame_queue.get()
            elif self.synthetic_view == "right":
                left_eye, right_eye, right_mask = self.frame_queue.get()
                left_eye = left_eye.clone()
            elif self.synthetic_view == "left":
                left_eye, right_eye, left_mask = self.frame_queue.get()
                right_eye = right_eye.clone()

            left_eye, right_eye = self._inpaint(
                left_eye,
                right_eye,
                left_mask,
                right_mask,
                self.synthetic_view,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                base_width=base_width,
            )

            if flush:
                left_eye = left_eye[self.pre_padding :]
                right_eye = right_eye[self.pre_padding :]
                self.frame_queue.clear()
            else:
                if self.post_padding > 0:
                    left_eye = left_eye[self.pre_padding : -self.post_padding]
                    right_eye = right_eye[self.pre_padding : -self.post_padding]
                elif self.pre_padding > 0:
                    left_eye = left_eye[self.pre_padding :]
                    right_eye = right_eye[self.pre_padding :]
                self.frame_queue.remove(self.model_seq - (self.pre_padding + self.post_padding))
            return left_eye, right_eye
        else:
            return None, None

    def infer(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str = "both",
        max_width: int | None = None,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert x.shape[0] <= self.model_seq
        x = self._resize(x, max_width)

        self.synthetic_view = synthetic_view
        self.base_width = depth.shape[-1]
        if self.frame_queue is None:
            self.frame_queue = self.create_frame_queue(x, depth, synthetic_view=synthetic_view)
        assert self.frame_queue is not None

        left_eye, right_eye, left_mask, right_mask = self.apply_warp(
            x,
            depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
        )

        for i in range(left_eye.shape[0]):
            repeat = self.pre_padding + 1 if self.frame_queue.empty() else 1
            if synthetic_view == "both":
                assert left_mask is not None
                assert right_mask is not None
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask[i], right_mask[i])
            elif synthetic_view == "right":
                assert right_mask is not None
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], right_mask=right_mask[i])
            elif synthetic_view == "left":
                assert left_mask is not None
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask=left_mask[i])

        return self(inner_dilation=inner_dilation, outer_dilation=outer_dilation, base_width=self.base_width)

    def flush(
        self, inner_dilation: int = 0, outer_dilation: int = 0, enable_amp: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.base_width is not None
        if self.frame_queue is None or self.frame_queue.empty():
            return None, None

        pad = self.frame_queue.fill()
        left_eye, right_eye = self(
            flush=True,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=self.base_width,
        )

        if pad > 0:
            return left_eye[:-pad], right_eye[:-pad]
        else:
            return left_eye, right_eye


class BaseInpaint(nn.Module):
    def __init__(self, device_id: int, image_model: InpaintComponent, video_model: InpaintComponent):
        super().__init__()
        self.device = create_device(device_id)
        self.model = nn.ModuleList([image_model, video_model])
        self.mode = 0
        self.to(self.device)
        self.eval()

    def set_mode(self, mode: str) -> None:
        assert mode in {"video", "image"}
        if mode == "video":
            self.mode = 1
        else:
            self.mode = 0

    def reset(self) -> None:
        cast(InpaintComponent, self.model[self.mode]).reset()

    def compile(self) -> None:
        self.model[self.mode].compile()

    def clear_compiled_model(self) -> None:
        cast(BaseVideoInpaint, self.model[self.mode]).clear_compiled_model()

    def compile_context(self, enabled: bool = True) -> contextlib.AbstractContextManager:
        if enabled:
            return CompileContext(self)
        else:
            return contextlib.nullcontext()

    @torch.inference_mode()
    def infer(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        synthetic_view: str = "both",
        max_width: int | None = None,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        preserve_screen_border: bool = False,
        enable_amp: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        with autocast(device=self.device, enabled=enable_amp):
            return cast(InpaintComponent, self.model[self.mode]).infer(
                x,
                depth,
                divergence=divergence,
                convergence=convergence,
                synthetic_view=synthetic_view,
                max_width=max_width,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                preserve_screen_border=preserve_screen_border,
                enable_amp=enable_amp,
            )

    @torch.inference_mode()
    def flush(
        self, inner_dilation: int = 0, outer_dilation: int = 0, enable_amp: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        with autocast(device=self.device, enabled=enable_amp):
            return cast(InpaintComponent, self.model[self.mode]).flush(
                inner_dilation=inner_dilation, outer_dilation=outer_dilation, enable_amp=enable_amp
            )
