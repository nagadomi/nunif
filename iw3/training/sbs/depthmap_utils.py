import torch
import torchvision.transforms.functional as TF
import numpy as np
from .stereoimage_generation import create_stereoimages


def force_update_midas_model():
    # Triggers fresh download of MiDaS repo
    torch.hub.help("isl-org/MiDaS", "DPT_BEiT_L_384", force_reload=True)


def load_zoed_model(device, model_type="ZoeD_N"):
    model = torch.hub.load("isl-org/ZoeDepth:main", model_type, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


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


def generate_sbs(model, im, divergence=2, convergence=1):
    im_org = im
    with torch.inference_mode():
        im_depth = model.infer_pil(im, output_type="pil")
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
