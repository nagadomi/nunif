import math
import torch
import torch.nn.functional as F
from .. models import get_model_config, get_model_device

def _calc_param(x, scale, offset, tile_size):
    p = {}
    p["x_h"] = x.shape[1]
    p["x_w"] = x.shape[2]
    input_offset = math.ceil(offset / scale)
    process_size = tile_size - input_offset * 2
    h_blocks = math.floor(p["x_h"] / process_size) + (0 if p["x_h"] % process_size == 0 else 1)
    w_blocks = math.floor(p["x_w"] / process_size) + (0 if p["x_w"] % process_size == 0 else 1)
    h = (h_blocks * process_size) + input_offset * 2
    w = (w_blocks * process_size) + input_offset * 2

    p["pad"] = (input_offset, (w - input_offset) - p["x_w"], input_offset, (h - input_offset) - p["x_h"])
    p["z_h"] = math.floor(p["x_h"] * scale)
    p["z_w"] = math.floor(p["x_w"] * scale)

    return p


def tiled_render(x, model, tile_size=256, batch_size=4):
    scale = get_model_config(model, "i2i_scale")
    offset = get_model_config(model, "i2i_offset")
    device = get_model_device(model)
    p = _calc_param(x, scale, offset, tile_size)
    x = F.pad(x.unsqueeze(0), p["pad"], mode='replicate')[0]
    ch, h, w = x.shape
    new_x = torch.zeros((ch, h * scale, w * scale)).to(device)

    output_size = tile_size * scale - offset * 2
    output_size_in_input = tile_size - math.ceil(offset / scale) * 2

    minibatch_index = 0
    minibatch = torch.zeros((batch_size, ch, tile_size, tile_size))
    output_indexes = [None] * batch_size

    for i in range(0, h, output_size_in_input):
        for j in range(0, w, output_size_in_input):
            if i + tile_size <= h and j + tile_size <= w:
                ii = i * scale
                jj = j * scale
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
