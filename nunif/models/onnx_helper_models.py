# helper models for onnxruntime-web
import copy

import onnx
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..logger import logger
from ..modules.reflection_pad2d import reflection_pad2d_loop
from ..utils.alpha import ChannelWiseSum
from .model import I2IBaseModel


class ONNXReflectionPadding(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(
        self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor
    ):
        return reflection_pad2d_loop(x, (int(left), int(right), int(top), int(bottom)))

    def export_onnx(self, f, **kwargs):
        """
        const ses = await ort.InferenceSession.create('./pad.onnx');
        var offset = BigInt(model_offset / model_scale);
        var pad = new ort.Tensor('int64', BigInt64Array.from([offset]), []);
        var out = await ses.run({"x": x, "left": pad, "right": pad, "top": pad, "bottom": pad});
        """
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        pad = torch.tensor(16, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        # ScriptModule requires dynamo=False
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x, pad, pad, pad, pad),
            f,
            input_names=["x", "left", "right", "top", "bottom"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 2: "height", 3: "width"},
            },
            **kwargs,
        )


class ONNXReplicationPadding(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(
        self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor
    ):
        return F.pad(x, (int(left), int(right), int(top), int(bottom)), mode="replicate")

    def export_onnx(self, f, **kwargs):
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        pad = torch.tensor(16, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x, pad, pad, pad, pad),
            f,
            input_names=["x", "left", "right", "top", "bottom"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 2: "height", 3: "width"},
            },
            **kwargs,
        )


def _hflip(x):
    return torch.flip(x, (-1,))


def _vflip(x):
    return torch.flip(x, (-2,))


class ONNXTTASplit(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: torch.Tensor):
        if int(tta_level) == 2:
            hflip = _hflip(x)
            x = torch.cat([x, hflip], dim=0)
        elif int(tta_level) == 4:
            hflip = _hflip(x)
            vflip = _vflip(x)
            vhflip = _hflip(vflip)
            x = torch.cat([x, hflip, vflip, vhflip], dim=0)
        # tta_level=8 is not supported due to rot90 is not supported in some onnx versions

        return x

    def export_onnx(self, f, **kwargs):
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        tta_level = torch.tensor(2, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x, tta_level),
            f,
            input_names=["x", "tta_level"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "input_batch_size", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 2: "height", 3: "width"},
            },
            **kwargs,
        )


class ONNXTTAMerge(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: torch.Tensor):
        if int(tta_level) == 2:
            x = torch.clamp((x[0] + _hflip(x[1])).unsqueeze(0) / 2.0, 0.0, 1.0)
        elif int(tta_level) == 4:
            hflip = _hflip(x[1])
            vflip = _vflip(x[2])
            vhflip = _vflip(_hflip(x[3]))
            x = torch.clamp((x[0] + hflip + vflip + vhflip).unsqueeze(0) / 4.0, 0.0, 1.0)

        return x

    def export_onnx(self, f, **kwargs):
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        tta_level = torch.tensor(2, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x, tta_level),
            f,
            input_names=["x", "tta_level"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "input_batch_size", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 2: "height", 3: "width"},
            },
            **kwargs,
        )


class ONNXCreateSeamBlendingFilter(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, scale: torch.Tensor, offset: torch.Tensor, tile_size: torch.Tensor):
        out_channels = 3
        blend_size = 16  # FIXME: Allow variable
        model_output_size = int(tile_size * scale - offset * 2)
        inner_tile_size = model_output_size - blend_size * 2
        x = torch.ones((out_channels, inner_tile_size, inner_tile_size), dtype=torch.float32)
        for i in range(blend_size):
            value = 1 - (1 / (blend_size + 1)) * (i + 1)
            x = F.pad(x, (1, 1, 1, 1), mode="constant", value=value)

        return x

    def export_onnx(self, f, **kwargs):
        scale = torch.tensor(2, dtype=torch.int64)
        offset = torch.tensor(16, dtype=torch.int64)
        tile_size = torch.tensor(64, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (scale, offset, tile_size),
            f,
            input_names=["scale", "offset", "tile_size"],
            output_names=["y"],
            dynamic_axes={"y": {0: "channels", 1: "height", 2: "width"}},
            **kwargs,
        )


class ONNXAlphaBorderPadding(nn.Module):
    # original code at nunif/utils/alpha.py
    # make it work on onnx
    def __init__(self):
        super().__init__()
        self.sum_alpha = ChannelWiseSum(1, 3)
        self.sum_rgb = ChannelWiseSum(3, 3)
        self.eval()

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor, offset: torch.Tensor):
        # rgb: CHW, alpha: CHW
        rgb = rgb.clone()
        alpha = alpha.squeeze(0)
        mask = alpha.new_zeros(alpha.shape)
        mask[alpha > 0.0] = 1.0
        mask_nega = (mask - 1.0).abs_().unsqueeze(0).expand(rgb.shape)
        rgb *= mask
        i = torch.zeros((1,), dtype=torch.int64)
        while torch.any(i < offset):
            i += 1
            mask_weight = self.sum_alpha(mask)
            border = self.sum_rgb(rgb)
            border /= mask_weight + 1e-7
            border *= mask_nega
            rgb *= mask
            rgb += border
            mask = (mask_weight > 0.0).float()
            mask_nega = (mask - 1.0).abs_().unsqueeze(0).expand(rgb.shape)

        return rgb.clamp_(0.0, 1.0)

    def to_inference_model(self):
        net = copy.deepcopy(self)
        net.eval()
        return net

    def export_onnx(self, f, **kwargs):
        rgb = torch.zeros((3, 256, 256), dtype=torch.float32)
        alpha = torch.zeros((1, 256, 256), dtype=torch.float32)
        offset = torch.tensor(16, dtype=torch.int64)
        model = torch.jit.script(self.to_inference_model())
        kwargs = dict(dynamo=False, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (rgb, alpha, offset),
            f,
            input_names=["rgb", "alpha", "offset"],
            output_names=["y"],
            dynamic_axes={
                "rgb": {1: "input_height", 2: "input_width"},
                "alpha": {1: "input_height", 2: "input_width"},
                "y": {1: "height", 2: "width"},
            },
            **kwargs,
        )


class ONNXScale1x(I2IBaseModel):
    # identity module for alpha channel in denoise
    def __init__(self, offset):
        super().__init__({}, scale=1, offset=offset, in_channels=3)

    def forward(self, x: torch.Tensor):
        p = self.i2i_offset
        if p == 0:
            return x
        return x[:, :, p:-p, p:-p]

    def export_onnx(self, f, **kwargs):
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        model = self.to_inference_model()
        kwargs = dict(dynamo=True, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x,),
            f,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 2: "height", 3: "width"},
            },
            **kwargs,
        )


class ONNXResizeBicubic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale_factor: torch.Tensor):
        x = F.interpolate(x, scale_factor=float(scale_factor), mode="bicubic", align_corners=False, antialias=False)
        return x

    def export_onnx(self, f, **kwargs):
        kwargs["opset_version"] = 18
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        scale_factor = torch.tensor(0.75, dtype=torch.float32)
        model = torch.jit.script(self.eval())
        kwargs = dict(dynamo=True, external_data=False) | kwargs
        torch.onnx.export(
            model,
            (x, scale_factor),
            f,
            input_names=["x", "scale_factor"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch_size", 1: "channels", 2: "input_height", 3: "input_width"},
                "y": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            },
            **kwargs,
        )
        patch_resize_antialias(f, index=0)


def patch_resize_antialias(onnx_path, name=None, index=None):
    """
    PyTorch's onnx exporter does not support bicubic downscaling with antialias=True.
    However, it is supported in ONNX optset 18.
    So once exported with antialias=False,
    then fixed antialias=True with ONNX file patch.
    """
    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, 18)
    assert model.opset_import[0].version >= 18
    onnx.checker.check_model(model)
    hit = False
    resize_count = 0
    for node in model.graph.node:
        if node.op_type == "Resize":
            do_patch = False
            if name is None and index is None:
                do_patch = True
            elif name is not None and name == node.name:
                do_patch = True
            elif index is not None and index == resize_count:
                do_patch = True
            if do_patch:
                antialias = onnx.helper.make_attribute("antialias", 1)
                node.attribute.extend([antialias])
                hit = True
            resize_count += 1
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    if not hit:
        logger.warning(f"patch_resize_antialias: No Resize node: {onnx_path}: name={name}, index={index}")


def test_onnx_model(model, input_args, input_names):
    import os

    import numpy as np
    from onnx.reference import ReferenceEvaluator

    os.makedirs("tmp/onnx", exist_ok=True)
    onnx_path = f"tmp/onnx/{model.__class__.__name__}.onnx"
    print(f"Exporting {model.__class__.__name__} to {onnx_path}...")
    model.export_onnx(onnx_path)

    print(f"Checking {onnx_path} with ReferenceEvaluator...")
    sess = ReferenceEvaluator(onnx_path)
    feed_dict = {}
    for name, arg in zip(input_names, input_args):
        if isinstance(arg, torch.Tensor):
            feed_dict[name] = arg.numpy()
        else:
            feed_dict[name] = np.array(arg)

    outputs = sess.run(None, feed_dict)
    for i, out in enumerate(outputs):
        print(f"  Output {i} shape: {out.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "pad_reflect",
            "pad_replicate",
            "tta_split",
            "tta_merge",
            "blend_filter",
            "alpha_border",
            "scale1x",
            "resize_bicubic",
        ],
    )
    args = parser.parse_args()

    if args.model == "pad_reflect" or args.model is None:
        test_onnx_model(
            ONNXReflectionPadding(),
            [
                torch.rand((1, 3, 256, 256)),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
            ],
            ["x", "left", "right", "top", "bottom"],
        )

    if args.model == "pad_replicate" or args.model is None:
        test_onnx_model(
            ONNXReplicationPadding(),
            [
                torch.rand((1, 3, 256, 256)),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
            ],
            ["x", "left", "right", "top", "bottom"],
        )

    if args.model == "tta_split" or args.model is None:
        test_onnx_model(
            ONNXTTASplit(), [torch.rand((1, 3, 256, 256)), torch.tensor(2, dtype=torch.int64)], ["x", "tta_level"]
        )

    if args.model == "tta_merge" or args.model is None:
        test_onnx_model(
            ONNXTTAMerge(), [torch.rand([2, 3, 256, 256]), torch.tensor(2, dtype=torch.int64)], ["x", "tta_level"]
        )

    if args.model == "blend_filter" or args.model is None:
        test_onnx_model(
            ONNXCreateSeamBlendingFilter(),
            [
                torch.tensor(2, dtype=torch.int64),
                torch.tensor(16, dtype=torch.int64),
                torch.tensor(64, dtype=torch.int64),
            ],
            ["scale", "offset", "tile_size"],
        )

    if args.model == "alpha_border" or args.model is None:
        test_onnx_model(
            ONNXAlphaBorderPadding(),
            [torch.zeros([3, 256, 256]), torch.zeros((1, 256, 256)), torch.tensor(16, dtype=torch.int64)],
            ["rgb", "alpha", "offset"],
        )

    if args.model == "scale1x" or args.model is None:
        test_onnx_model(ONNXScale1x(offset=16), [torch.rand((1, 3, 256, 256))], ["x"])

    if args.model == "resize_bicubic" or args.model is None:
        test_onnx_model(
            ONNXResizeBicubic(),
            [torch.rand((1, 3, 256, 256)), torch.tensor(0.75, dtype=torch.float32)],
            ["x", "scale_factor"],
        )


if __name__ == "__main__":
    main()
