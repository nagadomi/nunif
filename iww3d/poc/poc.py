import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image


def psnr(im1, im2):
    mse = ((im1 - im2) ** 2).mean()
    return 10 * torch.log10(1. / (mse + 1e-6))


def load_image(filename):
    im = Image.open(filename)
    im.load()
    return im


def normalize_depth(depth, depth_min, depth_max):
    depth = depth.float()
    depth = 1. - ((depth - depth_min) / (depth_max - depth_min))
    return depth


def apply_divergence(c, depth, divergence, image_width, shift):
    w, h = c.shape[2], c.shape[1]
    depth = depth.squeeze(0)
    index_shift = (1. - depth ** 2) * (shift * divergence * 0.01 * (image_width / w))

    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    mesh_x = mesh_x - index_shift
    grid = torch.stack((mesh_x, mesh_y), 2)
    grid = grid.unsqueeze(0)
    z = F.grid_sample(c.unsqueeze(0), grid, mode="bicubic", padding_mode="border", align_corners=True)
    z = z.squeeze(0)
    z = torch.clamp(z, 0, 1)
    return z


def test(basename):
    print(f"------- {basename}")
    im_c = load_image(f"{basename}_C.png")
    im_depth = load_image(f"{basename}_D.png")
    im_l = load_image(f"{basename}_L.png")
    im_r = load_image(f"{basename}_R.png")
    depth_max = int(im_depth.text["sbs_depth_max"])
    depth_min = int(im_depth.text["sbs_depth_min"])
    original_image_width = int(im_depth.text["sbs_width"])
    divergence = float(im_depth.text["sbs_divergence"])

    print("metadata",
          {"divergence": divergence,
           "depth_min": depth_min,
           "depth_max": depth_max,
           "width": original_image_width})

    c = TF.to_tensor(im_c)
    l = TF.to_tensor(im_l)
    r = TF.to_tensor(im_r)
    depth = TF.to_tensor(im_depth)
    depth = normalize_depth(depth, depth_min, depth_max)

    print("PSNR C x L", psnr(c, l), "PSNR L x R", psnr(l, r))

    test_l = apply_divergence(c, depth, divergence, original_image_width, shift=-1)
    test_r = apply_divergence(c, depth, divergence, original_image_width, shift=1)
    print("PSNR testL x L", psnr(test_l, l))
    print("PSNR testR x R", psnr(test_r, r))

    TF.to_pil_image(test_l).save(f"grid_sample_{basename}_L.png")
    TF.to_pil_image(test_r).save(f"grid_sample_{basename}_R.png")


if __name__ == "__main__":
    test("easy")
    test("normal")
