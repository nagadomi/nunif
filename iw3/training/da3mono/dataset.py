from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
)
from nunif.utils.image_loader import ImageLoader
from ...base_depth_model import BaseDepthModel


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0], (size, size))
    results = []
    for im in images:
        results.append(TF.crop(im, i, j, h, w))

    return tuple(results)


def center_crop(size, *images):
    results = []
    for im in images:
        results.append(TF.center_crop(im, (size, size)))

    return tuple(results)


def load_image(da3_path):
    da2_path = da3_path.replace("_DA3.png", "_DA2.png")
    da3, _ = BaseDepthModel.load_depth(da3_path)
    da2, _ = BaseDepthModel.load_depth(da2_path)

    return da3, da2


class DA3MonoDataset(Dataset):
    def __init__(self, input_dir, size, training):
        super().__init__()
        self.size = size
        self.training = training
        self.files = [fn for fn in ImageLoader.listdir(input_dir) if fn.endswith("_DA3.png")]
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        da3, da2 = load_image(self.files[index])
        da2 = da2 / 200.0  # reduce abs value

        if self.training:
            da3_crop, da2_crop = random_crop(self.size, da3, da2)
        else:
            da3_crop, da2_crop = center_crop(self.size, da3, da2)

        return da3_crop, da2_crop, index
