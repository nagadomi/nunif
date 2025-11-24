import torch
import threading
from . import video as VU


class AutoCrop():
    def __init__(self, mode="black", mod=2, frame_variation_threshold=0.95):
        mode = mode.lower()
        mode in {
            "black_tb", "black_lr", "black"
            "flat_tb", "flat_lr", "flat"
        }
        self.mode = mode
        self.mod = mod
        self.frame_variation_threshold = frame_variation_threshold
        self.black_only = self.mode in {"black_tb", "black_lr", "black"}
        self.reset()

    def reset(self):
        self.border_count_tb = None
        self.border_count_lr = None
        self.frame_count = 0

    def update(self, frame):
        if frame.ndim == 4:
            for i in range(frame.shape[0]):
                self.update(frame[i])
            return

        assert frame.ndim == 3

        if self.mode in {"black_tb", "black", "flat_lr", "flat"}:
            mask = self.detect_tb(frame, black_only=self.black_only)
            if self.border_count_tb is None:
                self.border_count_tb = mask.int()
            else:
                assert self.border_count_tb.shape == mask.shape
                self.border_count_tb = self.border_count_tb + mask.int()

        if self.mode in {"black_lr", "black", "flat_lr", "flat"}:
            mask = self.detect_lr(frame, black_only=self.black_only)
            if self.border_count_lr is None:
                self.border_count_lr = mask.int()
            else:
                assert self.border_count_lr.shape == mask.shape
                self.border_count_lr = self.border_count_lr + mask.int()

        self.frame_count += 1

    def get_crop(self, frame_variation_threshold=None):
        frame_variation_threshold = frame_variation_threshold or self.frame_variation_threshold
        if self.frame_count == 0:
            return slice(None), slice(None)

        if self.mode in {"black_tb", "flat_tb"}:
            slice_tb = self.mask_to_crop_tb(self.border_count_tb / self.frame_count >= frame_variation_threshold)
            slice_tb = self.apply_mod(slice_tb, self.mod)
            slice_lr = slice(None)
        elif self.mode in {"black_lr", "flat_lr"}:
            slice_lr = self.mask_to_crop_lr(self.border_count_lr / self.frame_count >= frame_variation_threshold)
            slice_lr = self.apply_mod(slice_lr, self.mod)
            slice_tb = slice(None)
        elif self.mode in {"black", "flat"}:
            slice_tb = self.mask_to_crop_tb(self.border_count_tb / self.frame_count >= frame_variation_threshold)
            slice_tb = self.apply_mod(slice_tb, self.mod)
            slice_lr = self.mask_to_crop_lr(self.border_count_lr / self.frame_count >= frame_variation_threshold)
            slice_lr = self.apply_mod(slice_lr, self.mod)

        return slice_tb, slice_lr

    @classmethod
    def autocrop(cls, frame, mode="black", mod=2):
        mode = mode.lower()
        mode in {
            "black_tb", "black_lr", "black"
            "flat_tb", "flat_lr", "flat"
        }
        black_only = mode in {"black_tb", "black_lr", "black"}

        if mode in {"black_tb", "black", "flat_lr", "flat"}:
            mask_tb = cls.detect_tb(frame, black_only=black_only)

        if mode in {"black_lr", "black", "flat_lr", "flat"}:
            mask_lr = cls.detect_lr(frame, black_only=black_only)

        if mode in {"black_tb", "flat_tb"}:
            slice_tb = cls.mask_to_crop_tb(mask_tb)
            slice_tb = cls.apply_mod(slice_tb, mod)
            slice_lr = slice(None)
        elif mode in {"black_lr", "flat_lr"}:
            slice_lr = cls.mask_to_crop_lr(mask_lr)
            slice_lr = cls.apply_mod(slice_lr, mod)
            slice_tb = slice(None)
        elif mode in {"black", "flat"}:
            slice_tb = cls.mask_to_crop_tb(mask_tb)
            slice_tb = cls.apply_mod(slice_tb, mod)
            slice_lr = cls.mask_to_crop_lr(mask_lr)
            slice_lr = cls.apply_mod(slice_lr, mod)

        return slice_tb, slice_lr

    @staticmethod
    def apply_mod(slice_value, mod):
        start = slice_value.start
        stop = slice_value.stop
        if start is not None:
            if start % mod != 0:
                # next
                start = start + (mod - start % mod)
        if stop is not None:
            if stop % mod != 0:
                # prev
                stop = stop - stop % mod

        return slice(start, stop)

    @staticmethod
    def rgb_to_y(x, tv_range):
        # NOTE: It might be better not to convert to Y when mode==flat
        if x.ndim == 3:
            # CHW
            r = x[0:1, :, :]
            g = x[1:2, :, :]
            b = x[2:3, :, :]
        elif x.ndim == 4:
            r = x[:, 0:1, :, :]
            g = x[:, 1:2, :, :]
            b = x[:, 2:3, :, :]
        else:
            raise ValueError(f"unsupported ndim {x.ndim}")

        # Convert to Y
        y = r * 0.299 + g * 0.587 + b * 0.114
        if tv_range:
            # Clamp to the TV range
            y = y.clamp(min=16.0 / 255.0, max=235.0 / 255.0)

        return y

    @classmethod
    def detect_tb(cls, x, black_only):
        y = cls.rgb_to_y(x, tv_range=black_only)
        mean = y.mean(dim=-1, keepdim=True)
        if black_only:
            is_dark = (mean <= 32.0 / 255.0)
            is_flat = (y - mean).abs().amax(dim=-1, keepdim=True) < 16 / 255.0
            is_bar = is_dark & is_flat
            return is_bar
        else:
            is_flat = (y - mean).abs().amax(dim=-1, keepdim=True) < 16 / 255.0
            return is_flat

    @classmethod
    def detect_lr(cls, x, black_only):
        y = cls.rgb_to_y(x, tv_range=black_only)
        mean = y.mean(dim=-2, keepdim=True)
        if black_only:
            is_dark = (mean <= 32.0 / 255.0)
            is_flat = (y - mean).abs().amax(dim=-2, keepdim=True) < 16 / 255.0
            is_bar = is_dark & is_flat
            return is_bar
        else:
            is_flat = (y - mean).abs().amax(dim=-2, keepdim=True) < 16 / 255.0
            return is_flat

    @classmethod
    def mask_to_crop_tb(cls, mask):
        assert mask.ndim == 3 and mask.shape[0] == 1 and mask.shape[2] == 1

        non_border_index = torch.nonzero(~mask.view(-1), as_tuple=False)
        if non_border_index.numel() == mask.numel():
            return slice(None, None)

        top = non_border_index[0].item()
        if top <= 0:
            top = None

        bottom = non_border_index[-1].item() + 1
        if bottom >= mask.shape[1]:
            bottom = None

        return slice(top, bottom)

    @classmethod
    def mask_to_crop_lr(cls, mask):
        assert mask.ndim == 3 and mask.shape[0] == 1 and mask.shape[1] == 1

        non_border_index = torch.nonzero(~mask.flatten(), as_tuple=False)

        if non_border_index.numel() == mask.numel():
            return slice(None, None)

        left = non_border_index[0].item()
        if left <= 0:
            left = None

        right = non_border_index[-1].item() + 1
        if right >= mask.shape[2]:
            right = None

        return slice(left, right)


def detect_border(
        video_file,
        mode="horizontal",
        mod=2,
        max_frames=1000,
        min_interval_sec=0.0,
        vf="",
        device="cuda",
        batch_size=2,
        stop_event=None,
        suspend_event=None,
        tqdm_fn=None,
        tqdm_title=None,
):
    model = AutoCrop(mode=mode, mod=mod)
    stop_event = stop_event or threading.Event()

    def batch_callback(x):
        model.update(x)
        if model.frame_count > max_frames:
            stop_event.set()

    callback_pool = VU.FrameCallbackPool(
        batch_callback,
        batch_size=batch_size,
        device=device,
        max_workers=0,
    )
    VU.process_video_keyframes(
        video_file, callback_pool, vf=vf,
        min_interval_sec=min_interval_sec,
        stop_event=stop_event, suspend_event=suspend_event,
        tqdm_fn=tqdm_fn,
        title=tqdm_title or "Border Detection",
    )
    return model.get_crop()


def _bench():
    import time
    import random

    N = 1000
    # S = (1080, 1920)  # HD 990 FPS
    S = (2160, 3840)  # 4K 330 FPS
    device = "cuda"
    top = random.randint(1, int(S[0] * 0.25))
    bottom = S[0] - random.randint(0, int(S[0] * 0.25))
    left = random.randint(1, int(S[1] * 0.25))
    right = S[1] - random.randint(0, int(S[1] * 0.25))

    t = time.perf_counter()
    torch.cuda.synchronize()
    autocrop = AutoCrop(mode="BLACK", mod=1)
    for i in range(10):
        frame = torch.rand((3, *S), device=device)
        autocrop.update(frame)
    autocrop.reset()
    for i in range(N):
        frame = torch.rand((3, *S), device=device)
        frame[:, 0:top, :] = 0.0
        frame[:, bottom:, :] = 0.0
        frame[:, :, 0:left] = 0.0
        frame[:, :, right:] = 0.0
        autocrop.update(frame)

    h, w = autocrop.get_crop()
    torch.cuda.synchronize()
    print(1 / ((time.perf_counter() - t) / N), "FPS")
    print("test", top, bottom, left, right, h, w)
    assert top == h.start and bottom == h.stop and left == w.start and right == w.stop


def _run_test():
    import argparse
    import torchvision.transforms.functional as TF
    import torchvision.io as io

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input rgb file")
    parser.add_argument("--mode", type=str, choices=["black_tb", "black_lr", "black", "flat_tb", "flat_lr", "flat"],
                        default="black",
                        help="mode")
    parser.add_argument("--vf", type=str, default="", help="video filter")
    args = parser.parse_args()

    if args.input.lower().endswith((".jpg", ".png", ".jpeg")):
        x = io.read_image(args.input) / 255.0
        h, w = AutoCrop.autocrop(x, mode=args.mode)
        print(h, w)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, h, w] = True
        mask = ~mask
        x[mask] = torch.tensor([1.0, 0.0, 1.0], dtype=x.dtype, device=x.device).view(3, 1, 1).expand_as(x)[mask]
        TF.to_pil_image(x[0]).show()
    else:
        # video
        h, w = detect_border(
            args.input,
            mode=args.mode,
            vf=args.vf
        )
        print(h, w)


if __name__ == "__main__":
    # _bench()
    _run_test()
