import torch
import torch.nn.functional as F
import math

EPS = 1e-7


def _calc_param(model, x, tile_size):
    p = {}
    offset = model.offset
    p["x_h"] = x.shape[1]
    p["x_w"] = x.shape[2]
    scale = model.scale
    input_offset = math.ceil(offset / scale)
    process_size = tile_size - input_offset * 2
    h_blocks = math.floor(p["x_h"] / process_size) + (0 if p["x_h"] % process_size == 0 else 1)
    w_blocks = math.floor(p["x_w"] / process_size) + (0 if p["x_w"] % process_size == 0 else 1)
    h = (h_blocks * process_size) + input_offset * 2
    w = (w_blocks * process_size) + input_offset * 2

    p["pad"] = (input_offset, (w - input_offset) - p["x_w"], input_offset, (h - input_offset) - p["x_h"])
    p["z_h"] = math.floor(p["x_h"] * model.scale)
    p["z_w"] = math.floor(p["x_w"] * model.scale)

    return p


def sum2d(x, kernel):
    if x.dim() == 2:
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), weight=kernel, stride=1, padding=1).squeeze(0).squeeze(0)
    else:
        return F.conv2d(x.unsqueeze(0), weight=kernel, stride=1, padding=1).squeeze(0)


def make_alpha_border(rgb, alpha, offset):
    if alpha is None:
        return rgb
    rgb = rgb.clone()
    alpha = alpha[0]
    kernel = torch.ones((1, 1, 3, 3)).to(rgb.device)
    mask = torch.zeros(alpha.shape).to(rgb.device)
    mask[alpha > 0] = 1
    mask_nega = (mask - 1).abs_().byte()

    rgb[0][mask_nega] = 0
    rgb[1][mask_nega] = 0
    rgb[2][mask_nega] = 0

    for i in range(offset):
        mask_weight = sum2d(mask, kernel)
        border = rgb.new(rgb.shape)
        for j in range(3):
            border[j].copy_(sum2d(rgb[j], kernel))
            border[j] /= mask_weight + EPS
            rgb[j][mask_nega] = border[j][mask_nega]
        mask.copy_(mask_weight)
        mask[mask_weight > 0] = 1
        mask_nega = (mask - 1).abs_().byte()

    return rgb.clamp_(0, 1)


def tiled_render(x, model, device, tile_size=256, batch_size=4):
    p = _calc_param(model, x, tile_size)
    x = F.pad(x.unsqueeze(0), p["pad"], mode='replicate')[0]
    ch, h, w = x.shape
    new_x = torch.zeros((ch, h * model.scale, w * model.scale)).to(device)

    output_size = tile_size * model.scale - model.offset * 2
    output_size_in_input = tile_size - math.ceil(model.offset / model.scale) * 2

    minibatch_index = 0
    minibatch = torch.zeros((batch_size, ch, tile_size, tile_size))
    output_indexes = [None] * batch_size

    for i in range(0, h, output_size_in_input):
        for j in range(0, w, output_size_in_input):
            if i + tile_size <= h and j + tile_size <= w:
                ii = i * model.scale
                jj = j * model.scale
                minibatch[minibatch_index] = x[:, i:i + tile_size, j:j + tile_size]
                output_indexes[minibatch_index] = (slice(None, None), slice(ii, ii + output_size), slice(jj, jj + output_size))
                minibatch_index += 1
                if minibatch_index == batch_size:
                    z = model(minibatch.to(device))
                    for k in range(minibatch_index):
                        new_x[output_indexes[k]] = z[k]
                    minibatch_index = 0
    if minibatch_index > 0:
        z = model(minibatch[0:minibatch_index].to(device))
        for k in range(minibatch_index):
            new_x[output_indexes[k]] = z[k]

    return new_x[:, :p["z_h"], :p["z_w"]].contiguous()


def simple_render(x, model, device):
    minibatch = True
    if x.dim() == 3:
        x = x.unsqueeze(0)
        minibatch = False
    x = x.to(device)
    if model.offset > 0:
        input_offset = math.ceil(model.offset / model.scale)
        x = F.pad(x, (input_offset,) * 4, mode='replicate')
    z = model(x)
    if not minibatch:
        z = z.squeeze(0)
    return z
