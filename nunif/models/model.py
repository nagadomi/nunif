import torch.nn as nn


class Model(nn.Module):
    name = "nunif.Model"
    def __init__(self, name, in_channels, out_channels=None, scale=None, offset=None, input_size=None):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.offset = offset
        self.input_size = input_size
        self.kwargs = {}
        self.updated_at = None

    def register_kwargs(self, kwargs):
        for name, value in kwargs.items():
            self.kwargs[name] = value

    def __repr__(self):
        return (f"name: {self.name}\nkwargs: {self.kwargs}\n" +
                super(Model, self).__repr__())
