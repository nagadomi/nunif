import torch
import torch.nn.functional as F

from nunif.models.utils import compile_model

from .models import MonoBW
from .backward_warp import apply_divergence_monobw
from .base_inpaint import BaseImageInpaint, BaseInpaint, BaseVideoInpaint, FrameQueue
from .dilation import dilate_inner, dilate_outer, mask_closing
from .inpaint_utils import (
    load_image_inpaint_model,
    load_video_inpaint_model,
)


class MonoBWInpaintImage(BaseImageInpaint):
    def __init__(self, name: str | None = None, device_id: int = -1):
        super().__init__(load_image_inpaint_model(name, device_id=device_id), device_id=device_id)
        self.monobw = MonoBW().eval()

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
        results = apply_divergence_monobw(
            self.monobw,
            x,
            depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
            fix_screen_border_mask=1,  # Fix the uninpaintable side
            return_mask=True,
        )
        assert len(results) == 4
        left_eye, right_eye, left_mask, right_mask = results
        return left_eye, right_eye, left_mask, right_mask

    def preprocess_mask(
        self,
        mask: torch.Tensor,
        target_size: tuple[int, int] | torch.Size,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(mask, size=target_size, mode="nearest")
        mask = mask > 0
        mask = mask_closing(mask)
        mask = dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)
        mask = dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        return mask


class MonoBWInpaintVideo(BaseVideoInpaint):
    def __init__(
        self,
        name: str | None = None,
        pre_padding: int = 3,
        post_padding: int = 3,
        device_id: int = -1,
    ):
        super().__init__(
            load_video_inpaint_model(name, device_id=device_id),
            pre_padding=pre_padding,
            post_padding=post_padding,
            device_id=device_id,
        )
        self.monobw = MonoBW()

    def compile(self) -> None:
        self.model_backup = (self.model, self.monobw)
        self.model = compile_model(self.model)
        self.monobw = compile_model(self.monobw)

    def clear_compiled_model(self) -> None:
        if self.model_backup is not None:
            self.model, self.monobw = self.model_backup
        self.model_backup = None

    def create_frame_queue(self, x, depth, synthetic_view) -> FrameQueue:
        return FrameQueue(
            synthetic_view=synthetic_view,
            seq=self.model_seq,
            height=x.shape[-2],
            width=x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

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
        results = apply_divergence_monobw(
            self.monobw,
            x,
            depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
            fix_screen_border_mask=2,  # Fix the both sides
            return_mask=True,
        )
        assert len(results) == 4
        left_eye, right_eye, left_mask, right_mask = results
        return left_eye, right_eye, left_mask, right_mask

    def preprocess_mask(
        self,
        mask: torch.Tensor,
        target_size: tuple[int, int] | torch.Size,
        inner_dilation: int = 0,
        outer_dilation: int = 0,
        base_width: int | None = None,
    ) -> torch.Tensor:
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(mask, size=target_size, mode="nearest")
        mask = mask > 0
        mask = mask_closing(mask)
        mask = dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)
        mask = dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        return mask


class MonoBWInpaint(BaseInpaint):
    def __init__(self, name: str | None, device_id: int, overlap_frames: list[int] | None = None):
        if overlap_frames is None:
            overlap_frames = [3]
        if len(overlap_frames) == 1:
            pre_padding = post_padding = overlap_frames[0]
        elif len(overlap_frames) == 2:
            pre_padding = overlap_frames[0]
            post_padding = overlap_frames[1]
        else:
            raise ValueError("overlap_frames requires 1 or 2 values")

        image_model = MonoBWInpaintImage(name, device_id=device_id)
        video_model = MonoBWInpaintVideo(
            name,
            pre_padding=pre_padding,
            post_padding=post_padding,
            device_id=device_id,
        )
        super().__init__(device_id, image_model, video_model)


def _test_image():
    import time

    import torchvision.io as io
    import torchvision.transforms.functional as TF

    model = MonoBWInpaintImage().cuda()
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


def _test_video(pre_padding=3, post_padding=3):
    import torchvision.io as io

    model = MonoBWInpaintVideo(pre_padding=pre_padding, post_padding=post_padding).cuda()
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
                left_eye, right_eye = model.infer(
                    x, depth, divergence=2.0, convergence=1.0, synthetic_view=synthetic_view
                )
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
    _test_video(pre_padding=3, post_padding=3)
    _test_video(pre_padding=3, post_padding=0)
