import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def normalize_depth(depth, depth_min=None, depth_max=None):
    depth = depth.float()
    if depth_min is None:
        depth_min = depth.min()
        depth_max = depth.max()

    if depth_max - depth_min > 0:
        depth = 1. - ((depth - depth_min) / (depth_max - depth_min))
    else:
        depth = torch.zeros_like(depth)
    return torch.clamp(depth, 0., 1.)


def make_divergence_feature_value(divergence, convergence, image_width):
    # assert image_width <= 2048
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0

    return divergence_feature_value, convergence_feature_value


def make_input_tensor(c, depth16, divergence, convergence,
                      image_width, depth_min=None, depth_max=None,
                      mapper="pow2"):
    w, h = c.shape[2], c.shape[1]
    depth = normalize_depth(depth16.squeeze(0), depth_min, depth_max)
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = torch.full_like(depth, divergence_value)
    convergence_feat = torch.full_like(depth, convergence_value)
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    grid = torch.stack((mesh_x, mesh_y), 2)
    grid = grid.permute(2, 0, 1)  # CHW

    return torch.cat([
        c,
        depth.unsqueeze(0),
        divergence_feat.unsqueeze(0),
        convergence_feat.unsqueeze(0),
        grid,
    ], dim=0)


@torch.inference_mode()
def batch_infer(model, im):
    x = TF.to_tensor(im).unsqueeze(0).to(model.device)
    x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)

    pad_h = int((x.shape[2] * 0.5) ** 0.5 * 3)
    pad_w = int((x.shape[3] * 0.5) ** 0.5 * 3)
    x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")

    out = model(x)['metric_depth']
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]),
                            mode="bicubic", align_corners=False)
    if pad_h > 0:
        out = out[:, :, pad_h:-pad_h, :]
    if pad_w > 0:
        out = out[:, :, :, pad_w:-pad_w]

    return ((out[0] + torch.flip(out[1], dims=[2])) * 128).cpu().to(torch.int16)


def softplus01(depth):
    # smooth function of `(depth - 0.5) * 2 if depth > 0.5 else 0`
    return torch.log(1. + torch.exp(depth * 12.0 - 6.)) / 6.0


def get_mapper(name):
    # https://github.com/nagadomi/nunif/assets/287255/0071a65a-62ff-4928-850c-0ad22bceba41
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01
    elif name == "softplus2":
        return lambda x: softplus01(x) ** 2
    else:
        raise NotImplementedError()


def equirectangular_projection(c, device="cpu"):
    c = c.to(device)
    h, w = c.shape[1:]
    max_edge = max(h, w)
    output_size = max_edge + max_edge // 2
    pad_w = (output_size - w) // 2
    pad_h = (output_size - h) // 2
    c = TF.pad(c, (pad_w, pad_h, pad_w, pad_h),
               padding_mode="constant", fill=0)

    h, w = c.shape[1:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                          torch.linspace(-1, 1, w, device=device), indexing="ij")

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    cos_elevation = torch.cos(elevation)
    x = cos_elevation * torch.sin(azimuth)
    y = torch.sin(elevation)
    z = cos_elevation * torch.cos(azimuth)
    mesh_x = 0.6666 * x / z
    mesh_y = 0.6666 * y / z
    grid = torch.stack((mesh_x, mesh_y), 2)
    z = F.grid_sample(c.unsqueeze(0),
                      grid.unsqueeze(0),
                      mode="bicubic", padding_mode="zeros",
                      align_corners=True).squeeze(0)
    z = torch.clamp(z, 0, 1)

    return z
