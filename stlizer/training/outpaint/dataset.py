import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
import torchvision.transforms as T
import nunif.transforms.std as TS
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
import nunif.utils.superpoint as KU
import random


def outpaint_mask(x):
    C, H, W = x.shape

    mask = torch.zeros((1, H, W), dtype=x.dtype, device=x.device)
    if random.random() < 0.5:
        padding_size = 1
    else:
        padding_size = int(random.random() * max(H, W) * 0.1)

    mask = F.pad(mask, (-padding_size,) * 4)
    mask = F.pad(mask, (padding_size,) * 4, mode="constant", value=1.0)

    hw = torch.tensor((H, W), dtype=torch.float32)
    shift = (torch.randn((2,)) * hw * 0.1).tolist()
    if random.random() < 0.5:
        angle = (torch.randn((1,)) * 15.0).item()
        center = (torch.randn((2,)) * hw * 0.1 + hw / 2.0).tolist()
    else:
        angle = 0
        center = [H // 2, W // 2]

    mask = KU.apply_transform(
        mask, shift=shift, scale=1.0, angle=angle,
        center=center, padding_mode="border"
    )
    mask = (mask >= 0.5).to(x.dtype)

    return mask


def inpaint_mask(x):
    C, H, W = x.shape

    method = random.randint(0, 1)
    if method == 0:
        p = random.uniform(0.001, 0.03)
        if random.random() < 0.5:
            scale_h = random.randint(4, 64)
            scale_w = random.randint(4, 16)
        else:
            scale_h = random.randint(4, 16)
            scale_w = random.randint(4, 64)

        p = torch.empty((1, H // scale_h, W // scale_w), dtype=x.dtype, device=x.device).fill_(p)
        mask = torch.bernoulli(p)
        mask = TF.resize(mask, (H, W), interpolation=InterpolationMode.NEAREST)
        mask = TF.rotate(mask, angle=random.uniform(-45, 45))
    elif method == 1:
        mask = torch.zeros((1, int(H * 2 ** 0.5), int(W * 2 ** 0.5)), dtype=x.dtype)
        size = random.randint(2, int(max(H, W) * 0.2))
        mask[:, (mask.shape[1] // 2 - size // 2):(mask.shape[1] // 2 + size), :] = 1.0
        mask = TF.rotate(mask, angle=random.uniform(-180, 180))
        sh = random.randint(0, mask.shape[1] - H)
        sw = random.randint(0, mask.shape[2] - W)
        mask = mask[:, sh:sh + H, sw:sw + W]

    mask = (mask >= 0.5).to(x.dtype)

    return mask


def apply_random_mask(x):
    if random.random() < 0.8:
        mask = outpaint_mask(x)
    else:
        mask = inpaint_mask(x)

    x = x * (1 - mask)
    mask = mask.to(torch.bool)

    return x, mask


class OutpaintDataset(Dataset):
    def __init__(self, input_dir, model_offset, tile_size, training):
        super().__init__()
        self.training = training
        self.model_offset = model_offset
        self.tile_size = tile_size
        self.files = list(ImageLoader.listdir(input_dir))
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")

        if self.training:
            self.gt_transform = T.Compose([
                TS.RandomSRHardExampleCrop(self.tile_size),
                T.RandomApply([TS.RandomGrayscale()], 0.005),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.gt_transform = T.CenterCrop(self.tile_size)

        if not training:
            # mask cache on cpu
            x = torch.zeros((3, self.tile_size, self.tile_size), dtype=torch.float32)
            self.masks = []
            for i in range(128):
                if i % 5 == 0:
                    mask = inpaint_mask(x)
                else:
                    mask = outpaint_mask(x)
                self.masks.append(mask)

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im, _ = load_image_simple(self.files[index], color="rgb")
        im = self.gt_transform(im)
        x = TF.to_tensor(im)
        if self.training:
            x, mask = apply_random_mask(x)
        else:
            mask = self.masks[index % len(self.masks)]
            x = x * (1 - mask)
            mask = mask.to(torch.bool)

        y = TF.to_tensor(TF.crop(im, self.model_offset, self.model_offset,
                                 im.height - self.model_offset * 2,
                                 im.width - self.model_offset * 2))
        return x, mask, y, index


def _test():
    import torchvision.io as io
    import time

    src = io.read_image("cc0/320/dog.png") / 255.0
    for _ in range(4):
        x, mask = apply_random_mask(src)
        TF.to_pil_image(x).show()
        time.sleep(2)


if __name__ == "__main__":
    _test()
