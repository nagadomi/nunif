from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
from torchvision import transforms as T
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.transforms import pair as TP
import nunif.transforms as TS
import random


USE_WAND = True


if not USE_WAND:
    INTERPOLATION_MODES = (
        InterpolationMode.BOX,
        InterpolationMode.BILINEAR,
        InterpolationMode.LANCZOS,
        InterpolationMode.BICUBIC,
    )
    INTERPOLATION_MODE_WEIGHTS = (2/9, 2/9, 4/9, 1/9)  # noqa: E226

    class RandomDownscaleX():
        def __init__(self, interpolation=None):
            self.interpolation = interpolation

        def __call__(self, x, y):
            w, h = x.size
            assert (w % 2 == 0 and h % 2 == 0)
            if self.interpolation is None:
                interpolation = random.choices(INTERPOLATION_MODES, weights=INTERPOLATION_MODE_WEIGHTS, k=1)[0]
            else:
                interpolation = self.interpolation
            x = TF.resize(x, size=(w // 2, h // 2), interpolation=interpolation, antialias=True)
            return x, y
else:
    from nunif.transforms import image_magick as IM
    INTERPOLATION_MODES = (
        "box",
        "sinc",
        "catrom"
    )
    # INTERPOLATION_MODE_WEIGHTS = (4/9, 4/9, 1/9)  # noqa: E226
    INTERPOLATION_MODE_WEIGHTS = (1/3, 1/3, 1/3)  # noqa: E226

    class RandomDownscaleX():
        def __init__(self, interpolation=None):
            self.interpolation = interpolation

        def __call__(self, x, y):
            w, h = x.size
            assert (w % 2 == 0 and h % 2 == 0)
            if self.interpolation is None:
                interpolation = random.choices(INTERPOLATION_MODES, weights=INTERPOLATION_MODE_WEIGHTS, k=1)[0]
            else:
                interpolation = self.interpolation
            x = pil_io.to_tensor(x)
            x = IM.resize(x, size=(h // 2, w // 2), filter_type=interpolation, blur=1.0)
            x = pil_io.to_image(x)
            return x, y


class RandomJPEGX():
    def __init__(self, level):
        self.level = 0

    def __call__(self, x, y):
        return x, y


class Waifu2xDataset(Dataset):
    def __init__(self, input_dir, num_samples=10000):
        super(Waifu2xDataset, self).__init__()
        self.files = ImageLoader.listdir(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.num_samples = num_samples

    def __len__(self):
        return len(self.files)

    def worker_init(self, worker_id):
        pass

    def sampler(self):
        return RandomSampler(
            self,
            num_samples=self.num_samples,
            replacement=True)

    def __getitem__(self, index):
        return self.files[index]


class Waifu2xScale2xDataset(Waifu2xDataset):
    def __init__(self, input_dir,
                 model_offset,
                 tile_size=104, num_samples=10000,
                 da_jpeg_p=0, da_scale_p=0, da_chshuf_p=0,
                 eval=False):
        super().__init__(input_dir, num_samples=num_samples)
        self.training = not eval
        if self.training:
            y_min_size = tile_size * 2 + 16
            self.gt_transforms = T.Compose([
                T.RandomApply([TS.RandomDownscale(min_size=y_min_size)], p=da_scale_p),
                T.RandomApply([TS.RandomChannelShuffle()], p=da_chshuf_p),
                T.RandomApply([TS.RandomJPEG(min_quality=92, max_quality=99)], p=da_jpeg_p),
            ])
            self.transforms = TP.Compose([
                TP.RandomHardExampleCrop(size=y_min_size, samples=4),
                RandomDownscaleX(),
                TP.RandomFlip(),
                TP.CenterCrop(size=tile_size, y_scale=2, y_offset=model_offset),
            ])
        else:
            self.gt_transforms = TS.Identity()
            if USE_WAND:
                interpolation = "catrom"
            else:
                interpolation = InterpolationMode.BICUBIC
            self.transforms = TP.Compose([
                TP.CenterCrop(size=tile_size * 2 + 16),
                RandomDownscaleX(interpolation=interpolation),
                TP.CenterCrop(size=tile_size, y_scale=2, y_offset=model_offset),
            ])

    def __getitem__(self, index):
        filename = super().__getitem__(index)
        im, _ = pil_io.load_image_simple(filename, color="rgb")
        if im is None:
            raise RuntimeError(f"Unable to load image: {filename}")
        im = self.gt_transforms(im)
        x, y = self.transforms(im, im)
        return TF.to_tensor(x), TF.to_tensor(y)


def _test():
    dataset = Waifu2xScale2xDataset("./data/waifu2x/eval",
                                    model_offset=36, tile_size=256)
    print(f"len {len(dataset)}")
    x, y = dataset[0]
    print("getitem[0]", x.size, y.size)
    x.show()
    y.show()


if __name__ == "__main__":
    _test()
