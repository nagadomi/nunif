import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import load_model
from nunif.device import create_device, autocast
from .dilation import mask_closing, dilate_outer, dilate_inner
from .forward_warp import apply_divergence_forward_warp
from . import models  # noqa


VIDEO_MODEL_URL = "models/video_inpant_v4/inpaint.light_video_inpaint_v1.pth"
IMAGE_MODEL_URL = "models/inpant_v7_gan4_ffc6/inpaint.light_inpaint_v1.pth"


def forward_right(model, right_eye, right_mask, inner_dilation, outer_dilation, base_width):
    right_mask = right_mask > 0
    right_mask = mask_closing(right_mask)
    right_mask = dilate_outer(right_mask, n_iter=outer_dilation, base_width=base_width)
    right_mask = dilate_inner(right_mask, n_iter=inner_dilation, base_width=base_width)

    right_eye = model.infer(right_eye, right_mask)
    return right_eye


def forward_left(model, left_eye, left_mask, inner_dilation, outer_dilation, base_width):
    left_mask = left_mask > 0
    left_mask = mask_closing(left_mask)
    left_mask = dilate_outer(left_mask, n_iter=outer_dilation, base_width=base_width)
    left_mask = dilate_inner(left_mask, n_iter=inner_dilation, base_width=base_width)

    left_eye, left_mask = left_eye.flip(-1), left_mask.flip(-1)
    left_eye = model.infer(left_eye, left_mask)
    left_eye, left_mask = left_eye.flip(-1), left_mask.flip(-1)

    return left_eye


class ForwardInpaintImage(nn.Module):
    def __init__(self, device_id=-1):
        super().__init__()
        self.model, _ = load_model(IMAGE_MODEL_URL, device_ids=[device_id])
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
            inner_dilation=0, outer_dilation=0, inpaint_max_width=1920,
            **_kwargs
    ):
        return self.forward(x, depth, divergence=divergence, convergence=convergence,
                            synthetic_view=synthetic_view,
                            inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                            inpaint_max_width=inpaint_max_width)

    def forward(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, inpaint_max_width=1920,
    ):
        if x.shape[-1] > inpaint_max_width:
            new_w = inpaint_max_width
            new_h = int((inpaint_max_width / x.shape[-1]) * x.shape[-2])
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


class FrameQueue():
    def __init__(self, synthetic_view, seq, height, width, dtype, device):
        self.left_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        self.right_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        if synthetic_view == "both":
            self.left_mask = torch.zeros((seq, 1, height, width), dtype=dtype, device=device)
            self.right_mask = torch.zeros((seq, 1, height, width), dtype=dtype, device=device)
        elif synthetic_view == "right":
            self.right_mask = torch.zeros((seq, 1, height, width), dtype=dtype, device=device)
            self.left_mask = None
        elif synthetic_view == "left":
            self.left_mask = torch.zeros((seq, 1, height, width), dtype=dtype, device=device)
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
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "right":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "left":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone())
        while not self.full():
            pad += 1
            self.add(**frame)

        return pad

    def remove(self, n):
        if n > 0 and n < self.max_index:
            for i in range(n):
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


class ForwardInpaintVideo(nn.Module):
    def __init__(self, pre_padding=3, post_padding=3, device_id=-1):
        super().__init__()
        self.model_seq = 12
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.model, _ = load_model(VIDEO_MODEL_URL, device_ids=[device_id])
        self.frame_queue = None
        self.synthetic_view = None
        self.inner_dilation = None
        self.outer_dilation = None
        self.base_width = None
        self.eval()

    def train(self, mode=True):
        super().train(mode=False)

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
                right_eye = forward_left(self.model, right_eye, right_mask, **forward_kwargs)
            elif self.synthetic_view == "right":
                left_eye, right_eye, right_mask = self.frame_queue.get()
                right_eye = forward_right(self.model, right_eye, right_mask, **forward_kwargs)
                left_eye = left_eye.clone()
            elif self.synthetic_view == "left":
                left_eye, right_eye, left_mask = self.frame_queue.get()
                left_eye = forward_left(self.model, left_eye, left_mask, **forward_kwargs)
                right_eye = right_eye.clone()

            if flush and self.frame_queue.full():  # TODO
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
            inner_dilation=0, outer_dilation=0, inpaint_max_width=1920,
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
                    self.frame_queue.add(left_eye[i], right_eye[i], left_mask[i], right_mask[i])
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

    @torch.inference_mode()
    def infer(
            self, x, depth,
            divergence, convergence, synthetic_view="both",
            inner_dilation=0, outer_dilation=0, inpaint_max_width=1920,
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
                inpaint_max_width=inpaint_max_width,
                **_kwargs
            )

    @torch.inference_mode()
    def flush(self, enable_amp=True):
        with autocast(device=self.device, enabled=enable_amp):
            return self.model[self.mode].flush()


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
    # _test_image()
    _test_video()
