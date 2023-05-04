import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
from torchvision import transforms as T
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.transforms import pair as TP
from nunif.training.sampler import HardExampleSampler, MiningMethod
import nunif.transforms as TS
from nunif.transforms import image_magick as IM
from .jpeg_noise import (
    RandomJPEGNoiseX,
    choose_validation_jpeg_quality,
    add_jpeg_noise,
    shift_jpeg_block,
)
from .photo_noise import RandomPhotoNoiseX, add_validation_noise
from PIL.Image import Resampling


NEAREST_PREFIX = "__NEAREST_"
INTERPOLATION_MODES = (
    "box",
    "sinc",
    "lanczos",
    "triangle",
    "catrom",
    # "vision.bicubic_no_antialias",
)
INTERPOLATION_NEAREST = "box"
INTERPOLATION_BICUBIC = "catrom"
# INTERPOLATION_MODE_WEIGHTS = (1/3, 1/3, 1/6, 1/16, 1/3, 1/12)  # noqa: E226
INTERPOLATION_MODE_WEIGHTS = (1/3, 1/3, 1/6, 1/16, 1/3)  # noqa: E226


def _resize(im, size, filter_type, blur):
    if filter_type in {"box", "sinc", "lanczos", "triangle", "catrom"}:
        return IM.resize(im, size, filter_type, blur)
    elif filter_type == "vision.bicubic_no_antialias":
        return TF.resize(im, size, InterpolationMode.BICUBIC, antialias=False)
    else:
        raise ValueError(filter_type)


def resize(im, size, filter_type, blur, enable_step=False):
    if enable_step and filter_type != INTERPOLATION_NEAREST and random.uniform(0, 1) < 0.1:
        h, w = im.shape[1:]
        scale = h / size[0]
        step1_scale = random.uniform(1, scale)
        step1_h, step1_w = int(step1_scale * h), int(step1_scale * w)
        im = _resize(im, (step1_h, step1_w), filter_type, 1)
        im = _resize(im, size, filter_type, blur)
        return im
    else:
        return _resize(im, size, filter_type, blur)


def pil_resize(im, size, filter_type):
    if filter_type == "box":
        resample = Resampling.BOX
    elif filter_type == "catrom":
        resample = Resampling.BICUBIC
    elif filter_type == "catrom":
        resample = Resampling.BICUBIC
    elif filter_type in {"sinc", "lanczos"}:
        resample = Resampling.LANCZOS
    elif filter_type == "triangle":
        resample = Resampling.BILINEAR
    else:
        raise NotImplementedError()

    return im.resize(size, resample=resample)


class RandomDownscaleX():
    def __init__(self, scale_factor, blur_shift=0, resize_blur_p=0.1, interpolation=None, training=True):
        assert scale_factor in {2, 4, 8}
        self.interpolation = interpolation
        self.scale_factor = scale_factor
        self.blur_shift = blur_shift
        self.training = training
        self.resize_blur_p = resize_blur_p

    def __call__(self, x, y):
        w, h = x.size
        if self.scale_factor == 1:
            return x, y
        assert (w % self.scale_factor == 0 and h % self.scale_factor == 0)
        if self.interpolation is None:
            interpolation = random.choices(INTERPOLATION_MODES, weights=INTERPOLATION_MODE_WEIGHTS, k=1)[0]
            fixed_interpolation = False
        else:
            interpolation = self.interpolation
            fixed_interpolation = True
        if self.scale_factor in {2, 4}:
            x = pil_io.to_tensor(x)
            if not self.training:
                blur = 1 + self.blur_shift / 4
            elif random.uniform(0, 1) < self.resize_blur_p:
                blur = random.uniform(0.95 + self.blur_shift, 1.05 + self.blur_shift)
            else:
                blur = 1
            x = resize(x, size=(h // self.scale_factor, w // self.scale_factor),
                       filter_type=interpolation, blur=blur, enable_step=self.training or fixed_interpolation)
            x = pil_io.to_image(x)
        elif self.scale_factor == 8:
            # wand 8x downscale is very slow for some reason
            # and, 8x is not used directly, so use pil instead
            x = pil_resize(x, (h // self.scale_factor, w // self.scale_factor), interpolation)

        return x, y


class RandomUnsharpMask():
    def __init__(self):
        pass

    def __call__(self, x):
        x = pil_io.to_tensor(x)
        x = IM.random_unsharp_mask(x)
        x = pil_io.to_image(x)
        return x


class AntialiasX():
    def __init__(self):
        pass

    def __call__(self, x, y):
        W, H = x.size
        interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
        if random.uniform(0, 1) < 0.5:
            scale = 2
        else:
            scale = random.uniform(1.5, 2)
        x = TF.resize(x, (int(H * scale), int(W * scale)), interpolation=interpolation, antialias=True)
        x = TF.resize(x, (H, W), interpolation=InterpolationMode.BICUBIC, antialias=True)
        return x, y


class Waifu2xDatasetBase(Dataset):
    def __init__(self, input_dir, num_samples,
                 hard_example_history_size=6):
        super().__init__()
        self.files = ImageLoader.listdir(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.num_samples = num_samples
        self.hard_example_history_size = hard_example_history_size

    def create_sampler(self):
        return HardExampleSampler(
            torch.ones((len(self),), dtype=torch.double),
            num_samples=self.num_samples,
            method=MiningMethod.TOP10,
            history_size=self.hard_example_history_size,
            scale_factor=4.,
        )

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]


class Waifu2xDataset(Waifu2xDatasetBase):
    def __init__(self, input_dir,
                 model_offset,
                 scale_factor,
                 tile_size, num_samples=None,
                 da_jpeg_p=0, da_scale_p=0, da_chshuf_p=0, da_unsharpmask_p=0,
                 da_grayscale_p=0, da_color_p=0, da_antialias_p=0,
                 bicubic_only=False,
                 deblur=0, resize_blur_p=0.1,
                 noise_level=-1, style=None,
                 training=True):
        assert scale_factor in {1, 2, 4, 8}
        assert noise_level in {-1, 0, 1, 2, 3}
        assert style in {None, "art", "photo"}

        super().__init__(input_dir, num_samples=num_samples)
        self.training = training
        self.style = style
        self.noise_level = noise_level
        if self.training:
            if noise_level >= 0:
                jpeg_transform = RandomJPEGNoiseX(style=style, noise_level=noise_level, random_crop=True)
            else:
                jpeg_transform = TP.Identity()

            if style == "photo":
                rotate_transform = TP.RandomApply([
                    TP.RandomChoice([
                        TP.RandomSafeRotate(y_scale=scale_factor, angle_min=-45, angle_max=45),
                        TP.RandomSafeRotate(y_scale=scale_factor, angle_min=-11, angle_max=11)
                    ], p=[0.2, 0.8]),
                ], p=0.2)
            else:
                rotate_transform = TP.Identity()

            if style == "photo" and noise_level >= 0:
                photo_noise = RandomPhotoNoiseX(noise_level=noise_level)
                if noise_level == 3:
                    jpeg_transform = T.RandomChoice([
                        jpeg_transform,
                        RandomPhotoNoiseX(noise_level=noise_level, force=True)], p=[0.95, 0.05])
            else:
                photo_noise = TP.Identity()

            antialias = TP.RandomApply([AntialiasX()], p=da_antialias_p)

            if scale_factor > 1:
                if bicubic_only:
                    interpolation = INTERPOLATION_BICUBIC
                else:
                    interpolation = None  # random
                random_downscale_x = RandomDownscaleX(scale_factor=scale_factor,
                                                      interpolation=interpolation,
                                                      blur_shift=deblur, resize_blur_p=resize_blur_p)
                random_downscale_x_nearest = RandomDownscaleX(scale_factor=scale_factor,
                                                              interpolation=INTERPOLATION_NEAREST)
            else:
                random_downscale_x = TP.Identity()
                random_downscale_x_nearest = TP.Identity()

            # 8(max jpeg shift size) * 2(max jpeg shift count) * scale_factor
            y_min_size = tile_size * scale_factor + (8 * 2 * scale_factor)
            self.gt_transforms = T.Compose([
                T.RandomApply([TS.RandomDownscale(min_size=y_min_size)], p=da_scale_p),
                T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)],
                              p=da_color_p),
                T.RandomApply([TS.RandomChannelShuffle()], p=da_chshuf_p),
                T.RandomApply([RandomUnsharpMask()], p=da_unsharpmask_p),
                # TODO: maybe need to prevent color noise for grayscale
                T.RandomApply([T.RandomGrayscale(p=1)], p=da_grayscale_p),
                T.RandomApply([TS.RandomJPEG(min_quality=92, max_quality=99)], p=da_jpeg_p),
            ])
            self.transforms = TP.Compose([
                TP.RandomHardExampleCrop(size=y_min_size, samples=4),
                random_downscale_x,
                photo_noise,
                rotate_transform,
                antialias,
                jpeg_transform,
                TP.RandomFlip(),
                TP.RandomCrop(size=tile_size, y_scale=scale_factor, y_offset=model_offset),
            ])
            self.transforms_nearest = TP.Compose([
                random_downscale_x_nearest,
                jpeg_transform,
                TP.RandomHardExampleCrop(size=tile_size,
                                         y_scale=scale_factor,
                                         y_offset=model_offset,
                                         samples=4),
                TP.RandomFlip(),
            ])
        else:
            self.gt_transforms = TS.Identity()
            interpolation = INTERPOLATION_BICUBIC
            if scale_factor > 1:
                downscale_x = RandomDownscaleX(scale_factor=scale_factor,
                                               blur_shift=deblur,
                                               interpolation=interpolation,
                                               training=False)
                downscale_x_nearest = RandomDownscaleX(scale_factor=scale_factor,
                                                       interpolation=INTERPOLATION_NEAREST,
                                                       training=False)
            else:
                downscale_x = TP.Identity()
                downscale_x_nearest = TP.Identity()
            y_min_size = tile_size * scale_factor + (8 * 2 * scale_factor)
            self.transforms = TP.Compose([
                TP.CenterCrop(size=y_min_size),
                downscale_x,
            ])
            self.transforms_nearest = TP.Compose([
                downscale_x_nearest,
            ])
            self.x_jpeg_shift = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
            self.center_crop = TP.CenterCrop(size=tile_size, y_scale=scale_factor, y_offset=model_offset)

    def __getitem__(self, index):
        filename = super().__getitem__(index)
        im, _ = pil_io.load_image_simple(filename, color="rgb")
        if im is None:
            raise RuntimeError(f"Unable to load image: {filename}")
        if NEAREST_PREFIX in filename:
            x, y = self.transforms_nearest(im, im)
        else:
            im = self.gt_transforms(im)
            x, y = self.transforms(im, im)

        if not self.training:
            if self.noise_level >= 0:
                if self.style == "photo":
                    x = add_validation_noise(x, noise_level=self.noise_level, index=index)
                qualities, subsampling = choose_validation_jpeg_quality(
                    index=index, style=self.style, noise_level=self.noise_level)
                for i, quality in enumerate(qualities):
                    x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)
                    if len(qualities) > 1 and i != len(qualities) - 1:
                        x, y = shift_jpeg_block(x, y, self.x_jpeg_shift[index % len(self.x_jpeg_shift)])
            x, y = self.center_crop(x, y)

        return TF.to_tensor(x), TF.to_tensor(y), index


def _test():
    dataset = Waifu2xDataset("./data/waifu2x/eval",
                             model_offset=36, tile_size=256, scale_factor=2,
                             style="art", noise_level=3)
    print(f"len {len(dataset)}")
    x, y, i = dataset[0]
    print("getitem[0]", x.size, y.size)
    TF.to_pil_image(x).show()
    TF.to_pil_image(y).show()


def _test_photo_noise():
    import cv2
    dataset = Waifu2xDataset("./data/photo/eval",
                             model_offset=36, tile_size=256, scale_factor=2,
                             style="photo", noise_level=3)
    print(f"len {len(dataset)}")
    for x, y, *_ in dataset:
        x = pil_io.to_cv2(pil_io.to_image(x))
        y = pil_io.to_cv2(pil_io.to_image(y))
        cv2.imshow("x", x)
        cv2.imshow("y", y)
        c = cv2.waitKey(0)
        if c in {ord("q"), ord("x")}:
            break


if __name__ == "__main__":
    _test_photo_noise()
