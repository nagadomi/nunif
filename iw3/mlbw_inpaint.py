import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.device import create_device, autocast
from nunif.models.utils import compile_model
from .backward_warp import apply_divergence_nn_delta_weight, postprocess_hole_mask
from . import models  # noqa
from .inpaint_utils import (
    FrameQueue,
    CompileContext,
    load_image_inpaint_model,
    load_video_inpaint_model,
    load_mask_mlbw,
)


MASK_MLBW_THRESHOLD = 0.15


def forward_right(model, right_eye, right_mask, inner_dilation, outer_dilation, base_width):
    right_mask = postprocess_hole_mask(right_mask, target_size=right_eye.shape[-2:], threshold=MASK_MLBW_THRESHOLD,
                                       inner_dilation=inner_dilation, outer_dilation=outer_dilation)
    right_eye = model.infer(right_eye, right_mask)
    return right_eye


def forward_left(model, left_eye, left_mask, inner_dilation, outer_dilation, base_width):
    left_eye, left_mask = left_eye.flip(-1), left_mask.flip(-1)
    left_mask = postprocess_hole_mask(left_mask, target_size=left_eye.shape[-2:], threshold=MASK_MLBW_THRESHOLD,
                                      inner_dilation=inner_dilation, outer_dilation=outer_dilation)
    left_eye = model.infer(left_eye, left_mask)
    left_eye = left_eye.flip(-1)
    return left_eye


def apply_divergence(model, c, depth, divergence, convergence, mapper, preserve_screen_border, synthetic_view, enable_amp):
    if synthetic_view == "both":
        left_eye, left_mask = apply_divergence_nn_delta_weight(
            model, c, depth, divergence=divergence, convergence=convergence, steps=1,
            mapper=mapper, shift=-1,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
            return_mask=True,
        )
        right_eye, right_mask = apply_divergence_nn_delta_weight(
            model, c, depth, divergence=divergence, convergence=convergence, steps=1,
            mapper=mapper, shift=1,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
            return_mask=True,
        )
    elif synthetic_view == "right":
        left_eye, left_mask = c, None
        right_eye, right_mask = apply_divergence_nn_delta_weight(
            model, c, depth, divergence=divergence * 2, convergence=convergence, steps=1,
            mapper=mapper, shift=1,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
            return_mask=True,
        )
    elif synthetic_view == "left":
        left_eye, left_mask = apply_divergence_nn_delta_weight(
            model, c, depth, divergence=divergence * 2, convergence=convergence, steps=1,
            mapper=mapper, shift=-1,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
            return_mask=True,
        )
        right_eye, right_mask = c, None

    return left_eye, right_eye, left_mask, right_mask


class MLBWInpaintImage(nn.Module):
    def __init__(self, device_id=-1, mask_mlbw=None):
        super().__init__()
        self.model = load_image_inpaint_model(device_id=device_id)
        if mask_mlbw is None:
            mask_mlbw = load_mask_mlbw(device_id=device_id)
        self.mask_mlbw = mask_mlbw
        self.device = create_device(device_id)
        self.eval()

    def train(self, mode=True):
        super().train(mode=False)

    def reset(self):
        pass

    def flush(self, enable_amp=True):
        return None, None

    def infer(
            self, x, depth,
            divergence, convergence,
            mapper="none",
            preserve_screen_border=False,
            synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=None,
            enable_amp=True,
            **_kwargs
    ):
        return self.forward(
            x, depth, divergence=divergence, convergence=convergence,
            mapper=mapper,
            preserve_screen_border=preserve_screen_border,
            synthetic_view=synthetic_view,
            inner_dilation=inner_dilation, outer_dilation=outer_dilation,
            max_width=max_width,
            enable_amp=enable_amp
        )

    def forward(
            self, x, depth,
            divergence, convergence,
            mapper="none",
            preserve_screen_border=False,
            synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=None,
            enable_amp=True,
    ):
        if max_width is not None and x.shape[-1] > max_width:
            if max_width % 2 != 0:
                max_width += 1
            new_w = max_width
            new_h = int((max_width / x.shape[-1]) * x.shape[-2])
            if new_h % 2 != 0:
                new_h += 1
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)

        left_eye, right_eye, left_mask, right_mask = apply_divergence(
            self.mask_mlbw, x, depth,
            divergence=divergence, convergence=convergence,
            mapper=mapper,
            preserve_screen_border=preserve_screen_border,
            synthetic_view=synthetic_view,
            enable_amp=enable_amp,
        )
        forward_kwargs = dict(
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=depth.shape[-1]
        )
        with autocast(device=self.device, enabled=enable_amp):
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


class MLBWInpaintVideo(nn.Module):
    # TODO: Refactor with ForwardInpaintVideo
    def __init__(self, pre_padding=3, post_padding=3, device_id=-1, mask_mlbw=None):
        super().__init__()
        if mask_mlbw is None:
            mask_mlbw = load_mask_mlbw(device_id=device_id)
        self.mask_mlbw = mask_mlbw

        self.model_seq = 12
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.model = load_video_inpaint_model(device_id=device_id)
        self.frame_queue = None
        self.synthetic_view = None
        self.inner_dilation = None
        self.outer_dilation = None
        self.base_width = None
        self.device = create_device(device_id)
        self.eval()

    def train(self, mode=True):
        super().train(mode=False)

    def compile(self):
        self.model_backup = (self.model, self.mask_mlbw)
        self.model = compile_model(self.model)
        self.mask_mlbw = compile_model(self.mask_mlbw)

    def clear_compiled_model(self):
        self.model, self.mask_mlbw = self.model_backup
        self.model_backup = None

    def reset(self):
        self.frame_queue = None

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
            divergence, convergence,
            mapper="none",
            preserve_screen_border=False,
            synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=None,
            enable_amp=True,
            **_kwargs
    ):
        assert x.shape[0] <= self.model_seq  # Prevent self.frame_queue growth
        if max_width is not None and x.shape[-1] > max_width:
            if max_width % 2 != 0:
                max_width += 1
            new_w = max_width
            new_h = int((max_width / x.shape[-1]) * x.shape[-2])
            if new_h % 2 != 0:
                new_h += 1
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)

        self.synthetic_view = synthetic_view
        self.inner_dilation = inner_dilation
        self.outer_dilation = outer_dilation
        self.base_width = depth.shape[-1]

        if self.frame_queue is None:
            self.frame_queue = FrameQueue(synthetic_view=synthetic_view,
                                          seq=self.model_seq,
                                          height=x.shape[-2], width=x.shape[-1],
                                          mask_height=depth.shape[-2], mask_width=depth.shape[-1],
                                          dtype=x.dtype, device=x.device)

        left_eye, right_eye, left_mask, right_mask = apply_divergence(
            self.mask_mlbw, x, depth,
            divergence=divergence, convergence=convergence,
            mapper=mapper,
            preserve_screen_border=preserve_screen_border,
            synthetic_view=synthetic_view,
            enable_amp=enable_amp,
        )
        for i in range(left_eye.shape[0]):
            repeat = self.pre_padding + 1 if self.frame_queue.empty() else 1
            if synthetic_view == "both":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask[i], right_mask[i])
            elif synthetic_view == "right":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], right_mask=right_mask[i])
            elif synthetic_view == "left":
                for _ in range(repeat):
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask=left_mask[i])

        with autocast(device=self.device, enabled=enable_amp):
            left_eye, right_eye = self.forward()
        if left_eye is not None:
            return left_eye, right_eye
        else:
            return None, None

    def flush(self, enable_amp=True):
        if self.frame_queue.empty():
            return None, None

        pad = self.frame_queue.fill()
        with autocast(device=self.device, enabled=enable_amp):
            left_eye, right_eye = self.forward(flush=True)

        if pad > 0:
            return left_eye[:-pad], right_eye[:-pad]
        else:
            return left_eye, right_eye


class MLBWInpaint(nn.Module):
    def __init__(self, device_id):
        super().__init__()
        self.device = create_device(device_id)
        self.model = nn.ModuleList([MLBWInpaintImage(device_id=device_id), MLBWInpaintVideo(device_id=device_id)])
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
            divergence, convergence,
            mapper,
            preserve_screen_border=False,
            synthetic_view="both",
            inner_dilation=0, outer_dilation=0, max_width=None,
            enable_amp=True,
            **_kwargs,
    ):
        return self.model[self.mode].infer(
            x, depth,
            divergence=divergence,
            convergence=convergence,
            mapper=mapper,
            preserve_screen_border=preserve_screen_border,
            synthetic_view=synthetic_view,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            max_width=max_width,
            enable_amp=enable_amp,
            **_kwargs
        )

    @torch.inference_mode()
    def flush(self, enable_amp=True):
        ret = self.model[self.mode].flush(enable_amp=enable_amp)
        return ret


def _test_image():
    import torchvision.transforms.functional as TF
    import torchvision.io as io
    import time

    model = MLBWInpaintImage().cuda()
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

    model = MLBWInpaintVideo().cuda()
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
