import math
import torch.nn.functional as F
from .. models import get_model_config, get_model_device
from .. device import autocast
from .seam_blending import SeamBlending


def tiled_render(x, model, tile_size=256, batch_size=4, enable_amp=False):
    return SeamBlending.tiled_render(
        x, model,
        tile_size=tile_size, batch_size=batch_size, enable_amp=enable_amp)


def simple_render(x, model, enable_amp=False, offset=None):
    scale = get_model_config(model, "i2i_scale")
    if offset is None:
        offset = get_model_config(model, "i2i_offset")
    device = get_model_device(model)
    minibatch = True
    if x.dim() == 3:
        x = x.unsqueeze(0)
        minibatch = False
    x = x.to(device)
    if offset > 0:
        input_offset = math.ceil(offset / scale)
        x = F.pad(x, (input_offset,) * 4, mode='replicate')
    with autocast(device, enabled=enable_amp):
        z = model(x)
    if not minibatch:
        z = z.squeeze(0)
    return z
