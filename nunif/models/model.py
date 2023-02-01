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

    def to_inference_model(self):
        net = copy.deepcopy(self)
        net.eval()
        return net

    def to_script_module(self):
        net = self.to_inference_model()
        return torch.jit.script(net)

    def export_onnx(self, f, **kwargs):
        raise NotImplementedError()


class I2IBaseModel(Model):
    name = "nunif.i2i_base_model"

    def __init__(self, kwargs, scale, offset, in_channels=None, in_size=None, blend_size=None):
        super(I2IBaseModel, self).__init__(kwargs)
        self.i2i_scale = scale
        self.i2i_offset = offset
        self.i2i_in_channels = in_channels
        self.i2i_in_size = in_size
        self.i2i_blend_size = blend_size

    def get_config(self):
        config = dict(super().get_config())
        config.update({
            "i2i_scale": self.i2i_scale,
            "i2i_offset": self.i2i_offset,
            "i2i_in_channels": self.i2i_in_channels,
            "i2i_in_size": self.i2i_in_size,
            "i2i_blend_size": self.i2i_blend_size,
        })
        return config

    def export_onnx(self, f, **kwargs):
        x = torch.rand([1, self.i2i_in_channels, 256, 256], dtype=torch.float32)
        model = self.to_inference_model()
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={'x': {0: 'batch_size', 2: "height", 3: "width"},
                          'y': {0: 'batch_size', 2: "height", 3: "width"}},
            **kwargs
        )


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
