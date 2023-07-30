import math
import torch
import torch.nn.functional as F
from .. models import get_model_config, get_model_device
from .. device import autocast


class SeamBlending(torch.nn.Module):
    def __init__(self, x_shape, scale, offset, tile_size, blend_size):
        super().__init__()

        C, H, W = x_shape
        config = SeamBlending.create_config((H, W), scale, offset, tile_size, blend_size)
        pixels = torch.zeros((C, config["y_buffer_h"], config["y_buffer_w"]), dtype=torch.float32)
        if blend_size > 0:
            weights = torch.zeros((C, config["y_buffer_h"], config["y_buffer_w"]), dtype=torch.float32)
            blend_filter = SeamBlending.create_blend_filter(scale, offset, tile_size, blend_size, C)
        else:
            weights = None
            blend_filter = None

        self.register_buffer("pixels", pixels)
        self.register_buffer("weights", weights)
        self.register_buffer("blend_filter", blend_filter)
        self.output_tile_step = config["output_tile_step"]
        self.input_tile_step = config["input_tile_step"]
        self.h_blocks = config["h_blocks"]
        self.w_blocks = config["w_blocks"]
        self.y_h = config["y_h"]
        self.y_w = config["y_w"]
        self.pad = config["pad"]
        self.blend_size = blend_size

    def forward(self, x: torch.Tensor, i: int, j: int):
        return SeamBlending.update(
            x, self.blend_filter, self.output_tile_step, self.blend_size,
            self.pixels, self.weights, i, j)

    def get_output(self):
        return torch.clamp(self.pixels[:, 0:self.y_h, 0:self.y_w], 0., 1.)

    def clear(self):
        if self.weights is not None:
            self.weights.zero_()
        self.pixels.zero_()

    @staticmethod
    def tiled_render(x, model, tile_size=256, batch_size=4, enable_amp=True,
                     config_callback=None, preprocess_callback=None, input_callback=None):
        assert not torch.is_grad_enabled()
        C, H, W = x.shape if config_callback is None else config_callback(x)
        scale = get_model_config(model, "i2i_scale")
        offset = get_model_config(model, "i2i_offset")
        blend_size = get_model_config(model, "i2i_blend_size")
        if blend_size is None:
            blend_size = 0
        device = get_model_device(model)

        seam_blending = SeamBlending(x.shape, scale=scale,
                                     offset=offset, tile_size=tile_size,
                                     blend_size=blend_size).to(device)
        seam_blending.eval()
        minibatch_index = 0
        minibatch = torch.zeros((batch_size, C, tile_size, tile_size), device=x.device)
        output_indexes = [None] * batch_size

        if preprocess_callback is not None:
            x = preprocess_callback(x, seam_blending.pad)
        else:
            x = F.pad(x.unsqueeze(0), seam_blending.pad, mode='replicate')[0]
        for h_i in range(seam_blending.h_blocks):
            for w_i in range(seam_blending.w_blocks):
                i = h_i * seam_blending.input_tile_step
                j = w_i * seam_blending.input_tile_step
                if input_callback is not None:
                    minibatch[minibatch_index] = input_callback(x, i, i + tile_size, j, j + tile_size)
                else:
                    minibatch[minibatch_index] = x[:, i:i + tile_size, j:j + tile_size]
                output_indexes[minibatch_index] = (h_i, w_i)
                minibatch_index += 1
                if minibatch_index == batch_size:
                    with autocast(device, enabled=enable_amp):
                        z = model(minibatch.to(device))
                    for k in range(minibatch_index):
                        seam_blending(z[k], output_indexes[k][0], output_indexes[k][1])
                    minibatch_index = 0

        if minibatch_index > 0:
            with autocast(device, enabled=enable_amp):
                z = model(minibatch[0:minibatch_index].to(device))
            for k in range(minibatch_index):
                seam_blending(z[k], output_indexes[k][0], output_indexes[k][1])

        return seam_blending.get_output().contiguous()

    @staticmethod
    def create_config(x_size, scale, offset, tile_size, blend_size):
        x_h = x_size[0]
        x_w = x_size[1]

        input_offset = math.ceil(offset / scale)
        input_blend_size = math.ceil(blend_size / scale)

        input_tile_step = tile_size - (input_offset * 2 + input_blend_size)
        h_blocks = w_blocks = input_h = input_w = 0
        while input_h < x_h + input_offset * 2:
            input_h = h_blocks * input_tile_step + tile_size
            h_blocks += 1
        while input_w < x_w + input_offset * 2:
            input_w = w_blocks * input_tile_step + tile_size
            w_blocks += 1

        output_tile_step = input_tile_step * scale
        output_h = input_h * scale
        output_w = input_w * scale

        p = {}
        p["y_h"] = math.floor(x_h * scale)
        p["y_w"] = math.floor(x_w * scale)
        p["h_blocks"] = h_blocks
        p["w_blocks"] = w_blocks
        p["pad"] = (input_offset,
                    input_w - (x_w + input_offset),
                    input_offset,
                    input_h - (x_h + input_offset))
        p["y_buffer_h"] = output_h
        p["y_buffer_w"] = output_w
        p["input_tile_step"] = input_tile_step
        p["output_tile_step"] = output_tile_step

        return p

    @staticmethod
    def create_blend_filter(scale, offset, tile_size, blend_size, out_channels):
        model_output_size = tile_size * scale - offset * 2
        inner_tile_size = model_output_size - blend_size * 2
        x = torch.ones((out_channels, inner_tile_size, inner_tile_size), dtype=torch.float32)
        for i in range(blend_size):
            value = 1 - (1 / (blend_size + 1)) * (i + 1)
            x = F.pad(x, (1, 1, 1, 1), mode="constant", value=value)
        return x

    @staticmethod
    def update(output_x, blend_filter, step_size, blend_size, pixels, weights, i, j):
        C, H, W = output_x.shape
        index = (slice(None, None),
                 slice(step_size * i, step_size * i + H),
                 slice(step_size * j, step_size * j + W))
        if blend_size > 0:
            assert blend_filter.shape == output_x.shape
            old_weight = weights[index]
            next_weight = old_weight + blend_filter
            old_weight = old_weight / next_weight  # old_weight <= next_weight
            new_weight = 1 - old_weight
            pixels[index] = pixels[index] * old_weight + output_x * new_weight
            weights[index] += blend_filter

            return pixels[index]
        else:
            # No blending
            pixels[index] = output_x
            return output_x
