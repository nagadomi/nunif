import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from .stereoimage_generation import create_stereoimages
from ... import zoedepth_model as ZU


def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = 0xffff
    if depth_max - depth_min > 0:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape)
    img_output = out.astype("uint16")
    img_output = 0xffff - img_output
    return img_output


def to_input_depth(im_depth):
    return normalize_depth(TF.to_tensor(im_depth).squeeze(0).cpu().numpy())


def generate_sbs(model, im, divergence=2, convergence=1, flip_aug=True, enable_amp=False):
    im_org = im
    with torch.inference_mode():
        depth = ZU.batch_infer(model, im, flip_aug=flip_aug, enable_amp=enable_amp)
        im_depth = Image.fromarray(depth.squeeze(0).numpy().astype(np.uint16))
    sbs = create_stereoimages(
        np.array(im_org, dtype=np.uint8),
        to_input_depth(im_depth),
        divergence,
        modes=["left-right"],
        convergence=convergence
    )[0]
    assert sbs.width == im_org.width * 2 and sbs.height == im_org.height
    assert im_depth.width == im_org.width and im_depth.height == im_org.height

    return sbs, im_depth


if __name__ == "__main__":
    pass
