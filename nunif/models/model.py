import torch.nn as nn


class Model(nn.Module):
    def __init__(self, name, in_channels, out_channels=None, inner_scale=None, offset=None, input_size=None):
        super(Model, self).__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_scale = inner_scale
        self.offset = offset
        self.input_size = input_size
        self._kwargs = {}

    def register_kwargs(self, kwargs):
        for name, value in kwargs.items():
            self._kwargs[name] = value

    def __repr__(self):
        s = f"""name: {self.name}
in_channels: {self.in_channels}
out_channels: {self.out_channels}
inner_scale: {self.inner_scale}
offset: {self.offset}
input_size: {self.input_size}\n"""
        return s + super(Model, self).__repr__()
