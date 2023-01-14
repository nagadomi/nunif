import torch
import torch.nn as nn
import copy


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

    def get_config(self):
        return {}

    def __repr__(self):
        return (f"name: {self.name}\nkwargs: {self.kwargs}\n" +
                super(Model, self).__repr__())

    def to_script_module(self):
        net = copy.deepcopy(self)
        net.eval()
        return torch.jit.script(net)


class I2IBaseModel(Model):
    name = "nunif.i2i_base_model"

    def __init__(self, kwargs, scale, offset, in_channels=None, in_size=None):
        super(I2IBaseModel, self).__init__(kwargs)
        self.i2i_scale = scale
        self.i2i_offset = offset
        self.i2i_in_channels = in_channels
        self.i2i_in_size = in_size

    def get_config(self):
        config = dict(super().get_config())
        config.update({
            "i2i_scale": self.i2i_scale,
            "i2i_offset": self.i2i_offset,
            "i2i_in_channels": self.i2i_in_channels,
            "i2i_in_size": self.i2i_in_size
        })
        return config


class SoftmaxBaseModel(Model):
    name = "nunif.softmax_base_model"

    def __init__(self, kwargs, class_names):
        super().__init__(kwargs)
        self.softmax_class_names = class_names

    def get_config(self):
        config = dict(super().get_config())
        config.update({
            "softmax_class_names": self.softmax_class_names
        })
        return config
