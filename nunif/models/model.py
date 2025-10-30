import torch
import torch.nn as nn
import copy
from functools import lru_cache


class Model(nn.Module):
    name = "nunif.Model"

    def __init__(self, kwargs):
        super(Model, self).__init__()
        self.kwargs = {}
        self.updated_at = None
        self.register_kwargs(kwargs)

    def get_device(self):
        return next(self.parameters()).device

    def register_kwargs(self, kwargs):
        for name, value in kwargs.items():
            if name not in {"self", "__class__"}:
                self.kwargs[name] = value

    def get_kwargs(self):
        return self.kwargs

    def __repr__(self):
        return (f"name: {self.name}\nkwargs: {self.kwargs}\n" +
                super(Model, self).__repr__())

    def to_inference_model(self):
        net = copy.deepcopy(self)
        net.eval()
        return net

    def to_script_module(self):
        net = self.to_inference_model()
        return torch.jit.script(net)

    def export_onnx(self, f, **kwargs):
        raise NotImplementedError()


_tile_size_validators = {}


def _register_tile_size_validator(name, func):
    _tile_size_validators[name] = func


@lru_cache
def _find_valid_tile_size(name, base_tile_size):
    validator = _tile_size_validators.get(name, None)
    if validator is not None:
        tile_size = int(base_tile_size)
        while tile_size > 0:
            if validator(tile_size):
                return tile_size
            tile_size -= 1
        raise ValueError(f"Could not find valid tile size: tile_size={base_tile_size}")
    else:
        return int(base_tile_size)


class I2IBaseModel(Model):
    name = "nunif.i2i_base_model"

    def __init__(self, kwargs, scale, offset, in_channels=None, in_size=None, blend_size=None,
                 default_tile_size=256, default_batch_size=4):
        super(I2IBaseModel, self).__init__(kwargs)
        self.i2i_scale = scale
        self.i2i_offset = offset
        self.i2i_in_channels = in_channels
        self.i2i_in_size = in_size
        self.i2i_blend_size = blend_size
        self.i2i_default_tile_size = default_tile_size
        self.i2i_default_batch_size = default_batch_size

    def register_tile_size_validator(self, validator):
        _register_tile_size_validator(self.name, validator)

    def find_valid_tile_size(self, base_tile_size):
        if base_tile_size is None:
            base_tile_size = self.i2i_default_tile_size
        tile_size = _find_valid_tile_size(self.name, base_tile_size)
        return tile_size

    def export_onnx(self, f, dynamo=False, **kwargs):
        shape = [1, self.i2i_in_channels, self.i2i_default_tile_size, self.i2i_default_tile_size]
        x = torch.rand(shape, dtype=torch.float32)
        model = self.to_inference_model()
        if dynamo:
            torch.onnx.export(
                model,
                x,
                f,
                input_names=["x"],
                output_names=["y"],
                dynamic_shapes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"}},
                dynamo=True,
                external_data=False,
                **kwargs
            )
        else:
            torch.onnx.export(
                model,
                x,
                f,
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                              "y": {0: "batch_size", 2: "height", 3: "width"}},
                external_data=False,
                **kwargs
            )


class SoftmaxBaseModel(Model):
    name = "nunif.softmax_base_model"

    def __init__(self, kwargs, class_names):
        super().__init__(kwargs)
        self.softmax_class_names = class_names
