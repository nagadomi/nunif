import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
import nunif.transforms.pair as TP
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
import random


SRC_SIZE = 256
CROP_SIZE = 64
NO_CHANGE_RATE = 0.1


class GenerateTrainingPair():
    def __call__(self, im1, im2):
        # assume im1 == im2
        width, height = im1.size
        if random.uniform(0, 1) < 0.5:
            scale = 0.25
        else:
            scale = random.uniform(0.25, 1.0)
        new_width = round(width * scale)
        new_height = round(height * scale)
        antialias = TF.resize(im1, (new_height, new_width),
                              interpolation=InterpolationMode.BILINEAR, antialias=True)
        if random.uniform(0, 1) < NO_CHANGE_RATE:
            no_antialias = antialias
        else:
            no_antialias = TF.resize(im1, (new_height, new_width), interpolation=InterpolationMode.NEAREST)

        return no_antialias, antialias


class GenerateValidationPair():
    def __init__(self):
        self.no_change_flags = [False] * (100 - int(100 * NO_CHANGE_RATE)) + [True] * int(100 * NO_CHANGE_RATE)
        self.scales = [0.25 + i / 99.0 * 0.75 for i in range(100)]
        self.index = 0

    def __call__(self, im1, im2):
        no_change_flag = self.no_change_flags[self.index % len(self.no_change_flags)]
        if self.index % 2 == 0:
            scale = 0.25
        else:
            scale = self.scales[self.index % len(self.scales)]
        width, height = im1.size
        new_width = round(width * scale)
        new_height = round(height * scale)
        antialias = TF.resize(im1, (new_height, new_width),
                              interpolation=InterpolationMode.BILINEAR, antialias=True)
        if no_change_flag:
            no_antialias = antialias
        else:
            no_antialias = TF.resize(im1, (new_height, new_width), interpolation=InterpolationMode.NEAREST)

        return no_antialias, antialias


class MinMaxNormalize():
    def __call__(self, x1, x2):
        x1 = TF.to_tensor(x1)
        x2 = TF.to_tensor(x2)
        min_value, max_value = x1.amin(), x1.amax()  # use input
        if min_value == max_value:
            x1 = torch.zeros_like(x1)
            x2 = torch.zeros_like(x2)
        else:
            x1 = (x1 - min_value) / (max_value - min_value)
            x2 = (x2 - min_value) / (max_value - min_value)

        return x1, x2


class DepthAADataset(Dataset):
    def __init__(self, input_dir, model_offset, training):
        super().__init__()
        self.training = training
        self.model_offset = model_offset
        self.files = list(ImageLoader.listdir(input_dir))
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")

        if self.training:
            self.transform = TP.Compose([
                TP.RandomHardExampleCrop(SRC_SIZE),
                GenerateTrainingPair(),
                TP.RandomCrop(CROP_SIZE),
                MinMaxNormalize()  # + to_tensor
            ])
        else:
            self.generator = GenerateValidationPair()
            self.transform = TP.Compose([
                TP.CenterCrop(SRC_SIZE),
                self.generator,
                TP.CenterCrop(CROP_SIZE),
                MinMaxNormalize()  # + to_tensor
            ])

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im, _ = load_image_simple(self.files[index], color="rgb")
        im = TF.to_grayscale(im)
        if self.training:
            x, y = self.transform(im, im)
        else:
            self.generator.index = index
            x, y = self.transform(im, im)
        return x, y


def _test():
    import time

    src, _ = load_image_simple("cc0/320/dog.png", color="rgb")
    src = TF.to_grayscale(src)
    # gen = GenerateTrainingPair()
    gen = GenerateValidationPair()
    for i in range(4):
        gen.index = i
        x, y = gen(src, src)
        x.show()
        time.sleep(2)
        y.show()
        time.sleep(2)


if __name__ == "__main__":
    _test()
