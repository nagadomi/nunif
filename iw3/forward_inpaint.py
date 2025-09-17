import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.device import create_device, autocast
from nunif.models.utils import compile_model
from .dilation import mask_closing, dilate_outer, dilate_inner
from .forward_warp import apply_divergence_forward_warp
from . import models  # noqa
from .inpaint_utils import (
    FrameQueue,
    CompileContext,
    load_image_inpaint_model,
    load_video_inpaint_model,
)


def forward_right(model, right_eye, right_mask, inner_dilation, outer_dilation, base_width):
    right_mask = right_mask > 0
    right_mask = mask_closing(right_mask)
    right_mask = dilate_outer(right_mask, n_iter=outer_dilation, base_width=base_width)
    right_mask = dilate_inner(right_mask, n_iter=inner_dilation, base_width=base_width)
    right_eye = model.infer(right_eye, right_mask)

    return right_eye


def forward_left(model, left_eye, left_mask, inner_dilation, outer_dilation, base_width):
    left_mask = left_mask > 0
    # flip for right view base
    left_eye, left_mask = left_eye.flip(-1), left_mask.flip(-1)

    left_mask = mask_closing(left_mask)
    left_mask = dilate_outer(left_mask, n_iter=outer_dilation, base_width=base_width)
    left_mask = dilate_inner(left_mask, n_iter=inner_dilation, base_width=base_width)
    left_eye = model.infer(left_eye, left_mask)

    left_eye = left_eye.flip(-1)

    return left_eye


class ForwardInpaintImage(nn.Module):
    def __init__(self, device_id=-1):
        super().__init__()
        self.model = load_image_inpaint_model(device_id=device_id)
        self.eval()

    def train(self, mode=True):
        super().train(mode=False)

    def reset(self):
        pass

    def flush(self):
        return None, None

    def infer(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=1920,
            **_kwargs
    ):
        return self.forward(x, depth, divergence=divergence, convergence=convergence,
                            synthetic_view=synthetic_view,
                            inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                            max_width=max_width)

    def forward(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=1920,
    ):
        if x.shape[-1] > max_width:
            if max_width % 2 != 0:
                max_width += 1
            new_w = max_width
            new_h = int((max_width / x.shape[-1]) * x.shape[-2])
            if new_h % 2 != 0:
                new_h += 1
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)

        left_eye, right_eye, left_mask, right_mask = apply_divergence_forward_warp(
            x, depth,
            divergence=divergence, convergence=convergence,
            synthetic_view=synthetic_view, return_mask=True
        )
        forward_kwargs = dict(
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=depth.shape[-1]
        )
        if synthetic_view == "both":
            left_eye = forward_left(self.model, left_eye, left_mask, **forward_kwargs)
            right_eye = forward_right(self.model, right_eye, right_mask, **forward_kwargs)
            return left_eye, right_eye
        elif synthetic_view == "right":
            right_eye = forward_right(self.model, right_eye, right_mask, **forward_kwargs)
            return left_eye, right_eye
        elif synthetic_view == "left":
            left_eye = forward_left(self.model, left_eye, left_mask, **forward_kwargs)
            return left_eye, right_eye


class ForwardInpaintVideo(nn.Module):
    def __init__(self, pre_padding=3, post_padding=3, device_id=-1):
        super().__init__()
        self.model_seq = 12
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.model = load_video_inpaint_model(device_id=device_id)
        self.frame_queue = None
        self.synthetic_view = None
        self.inner_dilation = None
        self.outer_dilation = None
        self.base_width = None
        self.eval()

    def train(self, mode=True):
        super().train(mode=False)

    def compile(self):
        self.model_backup = self.model
        self.model = compile_model(self.model)

    def clear_compiled_model(self):
        self.model = self.model_backup
        self.model_backup = None

    def reset(self):
        if self.frame_queue is not None:
            self.frame_queue.clear()

    def forward(self, flush=False):
        if self.frame_queue.full():
            forward_kwargs = dict(
                inner_dilation=self.inner_dilation,
                outer_dilation=self.outer_dilation,
                base_width=self.base_width,
            )
            if self.synthetic_view == "both":
                left_eye, right_eye, left_mask, right_mask = self.frame_queue.get()
                left_eye = forward_left(self.model, left_eye, left_mask, **forward_kwargs)
                right_eye = forward_right(self.model, right_eye, right_mask, **forward_kwargs)
            elif self.synthetic_view == "right":
                left_eye, right_eye, right_mask = self.frame_queue.get()
                right_eye = forward_right(self.model, right_eye, right_mask, **forward_kwargs)
                left_eye = left_eye.clone()
            elif self.synthetic_view == "left":
                left_eye, right_eye, left_mask = self.frame_queue.get()
                left_eye = forward_left(self.model, left_eye, left_mask, **forward_kwargs)
                right_eye = right_eye.clone()

            if flush:
                left_eye = left_eye[self.pre_padding:]
                right_eye = right_eye[self.pre_padding:]
                self.frame_queue.clear()
            else:
                if self.post_padding > 0:
                    left_eye = left_eye[self.pre_padding:-self.post_padding]
                    right_eye = right_eye[self.pre_padding:-self.post_padding]
                elif self.pre_padding > 0:
                    left_eye = left_eye[self.pre_padding:]
                    right_eye = right_eye[self.pre_padding:]
                self.frame_queue.remove(self.model_seq - (self.pre_padding + self.post_padding))
            return left_eye, right_eye
        else:
            return None, None

    def infer(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=1920,
            batch_size=2,
            **_kwargs,
    ):
        assert x.shape[0] <= self.model_seq  # Prevent self.frame_queue growth

        self.synthetic_view = synthetic_view
        self.inner_dilation = inner_dilation
        self.outer_dilation = outer_dilation
        self.base_width = depth.shape[-1]

        if self.frame_queue is None:
            self.frame_queue = FrameQueue(synthetic_view=synthetic_view,
                                          seq=self.model_seq,
                                          height=x.shape[-2], width=x.shape[-1],
                                          dtype=x.dtype, device=x.device)

        left_eye, right_eye, left_mask, right_mask = apply_divergence_forward_warp(
            x, depth,
            divergence=divergence, convergence=convergence,
            synthetic_view=synthetic_view, return_mask=True
        )
        for i in range(left_eye.shape[0]):
            repeat = self.pre_padding + 1 if self.frame_queue.empty() else 1
            if synthetic_view == "both":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask=left_mask[i], right_mask=right_mask[i])
            elif synthetic_view == "right":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], right_mask=right_mask[i])
            elif synthetic_view == "left":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask=left_mask[i])

        left_eye, right_eye = self.forward()
        if left_eye is not None:
            return left_eye, right_eye
        else:
            return None, None

    def flush(self):
        if self.frame_queue.empty():
            return None, None

        pad = self.frame_queue.fill()
        left_eye, right_eye = self.forward(flush=True)

        if pad > 0:
            return left_eye[:-pad], right_eye[:-pad]
        else:
            return left_eye, right_eye


class ForwardInpaint(nn.Module):
    def __init__(self, device_id):
        super().__init__()
        self.device = create_device(device_id)
        self.model = nn.ModuleList([ForwardInpaintImage(device_id=device_id), ForwardInpaintVideo(device_id=device_id)])
        self.mode = 0
        self.to(self.device)
        self.eval()

    def set_mode(self, mode):
        assert mode in {"video", "image"}
        if mode == "video":
            self.mode = 1
        else:
            self.mode = 0

    def reset(self):
        self.model[self.mode].reset()

    def compile(self):
        if hasattr(self.model[self.mode], "compile"):
            self.model[self.mode].compile()
        else:
            pass

    def clear_compiled_model(self):
        if hasattr(self.model[self.mode], "clear_compiled_model"):
            self.model[self.mode].clear_compiled_model()
        else:
            pass

    def compile_context(self, enabled=True):
        if enabled:
            return CompileContext(self)
        else:
            return contextlib.nullcontext()

    @torch.inference_mode()
    def infer(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=1920,
            enable_amp=True,
            **_kwargs,
    ):
        with autocast(device=self.device, enabled=enable_amp):
            return self.model[self.mode].infer(
                x, depth,
                divergence=divergence,
                convergence=convergence,
                synthetic_view=synthetic_view,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                max_width=max_width,
                **_kwargs
            )

    @torch.inference_mode()
    def flush(self, enable_amp=True):
        with autocast(device=self.device, enabled=enable_amp):
            ret = self.model[self.mode].flush()
            self.reset()
            return ret


def _test_image():
    import torchvision.transforms.functional as TF
    import torchvision.io as io
    import time

    model = ForwardInpaintImage().cuda()
    x = io.read_image("cc0/320/dog.png") / 255.0
    depth = io.read_image("cc0/depth/dog.png") / 65536.0

    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()

    with torch.autocast(device_type="cuda"), torch.inference_mode():
        left_eye, right_eye = model.infer(x, depth, divergence=2.0, convergence=1.0, synthetic_view="right")

    TF.to_pil_image(right_eye[0]).show()

    time.sleep(2)
    with torch.autocast(device_type="cuda"), torch.inference_mode():
        left_eye, right_eye = model.infer(x, depth, divergence=2.0, convergence=1.0, synthetic_view="left")

    TF.to_pil_image(left_eye[0]).show()


def _test_video():
    import torchvision.io as io

    model = ForwardInpaintVideo().cuda()
    x = io.read_image("cc0/320/dog.png") / 255.0
    depth = io.read_image("cc0/depth/dog.png") / 65536.0

    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()
    synthetic_view = "right"

    with torch.autocast(device_type="cuda"), torch.inference_mode():
        for test_frame in (1, 6, 12, 19, 24):
            model.reset()
            print(f"*** try process frames={test_frame}")
            ret_count = 0
            for i in range(test_frame):
                x[:, :, 0:4] = i / 255
                left_eye, right_eye = model.infer(x, depth, divergence=2.0, convergence=1.0, synthetic_view=synthetic_view)
                if right_eye is None:
                    print("infer None")
                else:
                    print(f"infer {right_eye.shape}")
                    for left, right in zip(left_eye, right_eye):
                        print("frame", left[0, 0, 0] * 255, right[0, 0, 0] * 255)
                    ret_count += right_eye.shape[0]

            left_eye, right_eye = model.flush()
            if right_eye is None:
                print("flush None")
            else:
                print(f"flush {right_eye.shape}")
                for left, right in zip(left_eye, right_eye):
                    print("frame", left[0, 0, 0] * 255, right[0, 0, 0] * 255)
                ret_count += right_eye.shape[0]

            assert ret_count == test_frame
            print("OK")


if __name__ == "__main__":
    _test_image()
    _test_video()
