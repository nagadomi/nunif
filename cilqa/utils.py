import torch
import torchvision.transforms.functional as TF
from nunif.models import get_model_device


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


def predict_jpeg_quality(model, im, num_patches=8, patch_size=PATCH_SIZE):
    device = get_model_device(model)
    x = extract_patches(im, num_patches, patch_size)
    x = x.to(device)
    quality, subsampling = model(x)
    quality = torch.clamp(quality.mean(), 0, 100).item()
    subsampling_prob = torch.sigmoid(subsampling).mean().item()

    return quality, subsampling_prob
