import sys
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from nunif.models import get_model_device
from nunif.utils.image_loader import list_images
from nunif.utils import pil_io
from nunif.device import autocast


PATCH_SIZE = 128


def safe_pad(im, min_size):
    w, h = im.size
    if h < min_size or w < min_size:
        im = TF.pad(im, (0, 0, max(0, min_size - w), max(0, min_size - h)), padding_mode="reflect")
    return im


def extract_patches(im, num_patches, patch_size=PATCH_SIZE):
    stride = patch_size
    im = safe_pad(im, patch_size)
    im = TF.to_tensor(im)
    c, h, w = im.shape
    patches = []
    for y in range(0, h, stride):
        if not y + patch_size <= h:
            break
        for x in range(0, w, stride):
            if not x + patch_size <= w:
                break
            patch = im[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    # BCHW
    patches = torch.stack(patches)
    # select top-k high variance patch
    color_stdv = torch.std(patches, dim=[2, 3]).mean(dim=1)
    _, indexes = torch.topk(color_stdv, min(num_patches, color_stdv.shape[0]))
    patches = patches[indexes]
    return patches


def predict_jpeg_quality(model, x, num_patches=8, patch_size=PATCH_SIZE):
    device = get_model_device(model)
    if isinstance(x, Image.Image):
        x = extract_patches(x, num_patches, patch_size)
    x = x.to(device)
    with autocast(device):
        quality, subsampling = model(x)
    quality = torch.clamp(quality.mean(), 0, 100).item()
    subsampling_prob = torch.sigmoid(subsampling).mean().item()

    return quality, subsampling_prob


def predict_grain_noise_psnr(model, x, num_patches=8, patch_size=PATCH_SIZE):
    device = get_model_device(model)
    if isinstance(x, Image.Image):
        x = extract_patches(x, num_patches, patch_size)
    x = x.to(device)
    with autocast(device):
        noise_level = model(x)
    noise_level = torch.clamp(noise_level.mean(), 0, 50).item()
    psnr = 50. - noise_level

    return psnr


class PatchDataset(Dataset):
    def __init__(self, input_dir, num_patches=8):
        super().__init__()
        self.files = list_images(input_dir)
        self.num_patches = num_patches

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        im, meta = pil_io.load_image_simple(filename, color="rgb")
        if im is None:
            print(f"{filename}: Load Error", file=sys.stderr)
            return torch.zeros((self.num_patches, 3, PATCH_SIZE, PATCH_SIZE)), ""
        patch = extract_patches(im, self.num_patches, patch_size=PATCH_SIZE)
        return patch, filename


def create_patch_loader(input_dir, num_patches=8, num_workers=4):
    loader = DataLoader(
        PatchDataset(input_dir, num_patches=num_patches),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=8,
        drop_last=False
    )
    return loader
