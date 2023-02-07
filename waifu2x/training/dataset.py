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
from .noise_level import (
    RandomJPEGNoiseX,
    choose_validation_jpeg_quality,
    add_jpeg_noise,
    shift_jpeg_block,
)

NEAREST_PREFIX = "__NEAREST_"
INTERPOLATION_MODES = (
    "box",
    "sinc",
    "lanczos",
    "triangle",
    "catrom",
#    "vision.bicubic_no_antialias",
)
INTERPOLATION_NEAREST = "box"
#INTERPOLATION_MODE_WEIGHTS = (1/3, 1/3, 1/6, 1/16, 1/3, 1/12)  # noqa: E226
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


class RandomDownscaleX():
    def __init__(self, scale_factor, blur_shift=0, resize_blur_p=0.1, interpolation=None, training=True):
        assert scale_factor in {2, 4}
        self.interpolation = interpolation
        self.scale_factor = scale_factor
        self.blur_shift = blur_shift
        self.training = training
        self.resize_blur_p = resize_blur_p

    def __call__(self, x, y):
        w, h = x.size
        assert (w % self.scale_factor == 0 and h % self.scale_factor == 0)
        x = pil_io.to_tensor(x)
        if self.interpolation is None:
            interpolation = random.choices(INTERPOLATION_MODES, weights=INTERPOLATION_MODE_WEIGHTS, k=1)[0]
        else:
            interpolation = self.interpolation
        if self.scale_factor == 2:
            if not self.training:
                blur = 1 + self.blur_shift / 4
            elif random.uniform(0, 1) < self.resize_blur_p:
                blur = random.uniform(0.95 + self.blur_shift, 1.05 + self.blur_shift)
            else:
                blur = 1
            x = resize(x, size=(h // self.scale_factor, w // self.scale_factor),
                       filter_type=interpolation, blur=blur, enable_step=self.training)
        elif self.scale_factor == 4:
            if not self.training:
                blur = 1 + self.blur_shift / 4
            elif random.uniform(0, 1) < self.resize_blur_p:
                blur = random.uniform(0.95 + self.blur_shift, 1.05 + self.blur_shift)
            else:
                blur = 1
            x = resize(x, size=(h // self.scale_factor, w // self.scale_factor),
                       filter_type=interpolation, blur=blur, enable_step=self.training)
        x = pil_io.to_image(x)
        return x, y


class RandomUnsharpMask():
    def __init__(self):
        pass

    def __call__(self, x):
        x = pil_io.to_tensor(x)
        x = IM.random_unsharp_mask(x)
        x = pil_io.to_image(x)
        return x


class Waifu2xDatasetBase(Dataset):
    def __init__(self, input_dir, num_samples=None):
        super().__init__()
        self.files = ImageLoader.listdir(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.num_samples = num_samples
        if num_samples is not None:
            self._sampler = HardExampleSampler(
                torch.ones((len(self),), dtype=torch.double),
                num_samples=num_samples,
                method=MiningMethod.TOP10,
                history_size=6,
                scale_factor=4,
            )
        else:
            self._sampler = None

    def __len__(self):
        return len(self.files)

    def worker_init(self, worker_id):
        pass

    def sampler(self):
        return self._sampler

    def set_hard_example(self, method):
        if method == "top10":
            self._sampler.method = MiningMethod.TOP10
        elif method == "top20":
            self._sampler.method = MiningMethod.TOP20
        elif method == "linear":
            self._sampler.method = MiningMethod.LINEAR

    def update_hard_example_losses(self, indexes, loss):
        self._sampler.update_losses(indexes, loss)

    def update_hard_example_weights(self):
        self._sampler.update_weights()

    def __getitem__(self, index):
        return self.files[index]


class Waifu2xDataset(Waifu2xDatasetBase):
    def __init__(self, input_dir,
                 model_offset,
                 scale_factor,
                 tile_size, num_samples=None,
                 da_jpeg_p=0, da_scale_p=0, da_chshuf_p=0, da_unsharpmask_p=0, da_grayscale_p=0,
                 deblur=0, resize_blur_p=0.1,
                 noise_level=-1, style=None,
                 training=True):
        assert scale_factor in {1, 2, 4}
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
            if scale_factor > 1:
                random_downscale_x = RandomDownscaleX(scale_factor=scale_factor,
                                                      blur_shift=deblur, resize_blur_p=resize_blur_p)
                random_downscale_x_nearest = RandomDownscaleX(scale_factor=scale_factor,
                                                              interpolation=INTERPOLATION_NEAREST)
            else:
                random_downscale_x = TP.Identity()
                random_downscale_x_nearest = TP.Identity()

            # 64 = 8(max jpeg shift size) * 4(max_scale_factor) * 2(max jpeg shift count)
            y_min_size = tile_size * scale_factor + 64
            self.gt_transforms = T.Compose([
                T.RandomApply([TS.RandomDownscale(min_size=y_min_size)], p=da_scale_p),
                T.RandomApply([TS.RandomChannelShuffle()], p=da_chshuf_p),
                T.RandomApply([RandomUnsharpMask()], p=da_unsharpmask_p),
                T.RandomApply([T.RandomGrayscale(p=1)], p=da_grayscale_p),
                T.RandomApply([TS.RandomJPEG(min_quality=92, max_quality=99)], p=da_jpeg_p),
            ])
            self.transforms = TP.Compose([
                TP.RandomHardExampleCrop(size=y_min_size, samples=4),
                random_downscale_x,
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
            interpolation = "catrom"
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
            y_min_size = tile_size * scale_factor + 64
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


if __name__ == "__main__":
    _test()
