# helper models for onnxruntime-web
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import onnx
import copy
from .model import I2IBaseModel
from ..modules.reflection_pad2d import reflection_pad2d_loop
from ..utils.alpha import ChannelWiseSum
from ..logger import logger


class ONNXReflectionPadding(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, left: int, right: int, top: int, bottom: int):
        return reflection_pad2d_loop(x, (left, right, top, bottom))

    def export_onnx(self, f, **kwargs):
        """
         const ses = await ort.InferenceSession.create('./pad.onnx');
         var offset = BigInt(model_offset / model_scale);
         var pad = new ort.Tensor('int64', BigInt64Array.from([offset]), []);
         var out = await ses.run({"x": x, "left": pad, "right": pad, "top": pad, "bottom": pad});
        """
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        pad = [512, 120, 512, 120]
        model = torch.jit.script(self.to_inference_model())
        torch.onnx.export(
            model,
            [x, *pad],
            f,
            input_names=["x", "left", "right", "top", "bottom"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXReplicationPadding(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, left: int, right: int, top: int, bottom: int):
        return F.pad(x, (left, right, top, bottom), mode="replicate")

    def export_onnx(self, f, **kwargs):
        """
         const ses = await ort.InferenceSession.create('./pad.onnx');
         var offset = BigInt(model_offset / model_scale);
         var pad = new ort.Tensor('int64', BigInt64Array.from([offset]), []);
         var out = await ses.run({"x": x, "left": pad, "right": pad, "top": pad, "bottom": pad});
        """
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        pad = 4
        model = torch.jit.script(self.to_inference_model())
        torch.onnx.export(
            model,
            [x, pad, pad, pad, pad],
            f,
            input_names=["x", "left", "right", "top", "bottom"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )


def _hflip(x):
    return torch.flip(x, (-1,))


def _vflip(x):
    return torch.flip(x, (-2,))


class ONNXTTASplit(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: int):
        if tta_level == 2:
            hflip = _hflip(x)
            x = torch.cat([x, hflip], dim=0)
        elif tta_level == 4:
            hflip = _hflip(x)
            vflip = _vflip(x)
            vhflip = _hflip(vflip)
            x = torch.cat([x, hflip, vflip, vhflip], dim=0)
        # tta_level=8 is not supported due to rot90 is not supported

        return x

    def export_onnx(self, f, **kwargs):
        """
         const ses = await ort.InferenceSession.create('./tta_split.onnx');
         var tta_level = 2;
         var tta_level = BigInt(tta_level);
         var out = await ses.run({"x": x, "tta_level": tta_level});
        """
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        tta_level = 2
        model = torch.jit.script(self.to_inference_model())
        torch.onnx.export(
            model,
            [x, tta_level],
            f,
            input_names=["x", "tta_level"],
            output_names=["y"],
            dynamic_axes={"x": {0: "input_batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXTTAMerge(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: int):
        if tta_level == 2:
            x = torch.clamp((x[0] + _hflip(x[1])).unsqueeze(0) / 2., 0., 1.)
        elif tta_level == 4:
            hflip = _hflip(x[1])
            vflip = _vflip(x[2])
            vhflip = _vflip(_hflip(x[3]))
            x = torch.clamp((x[0] + hflip + vflip + vhflip).unsqueeze(0) / 4., 0., 1.)

        return x

    def export_onnx(self, f, **kwargs):
        """
         const ses = await ort.InferenceSession.create('./tta_merge.onnx');
         var tta_level = 2;
         var tta_level = BigInt(tta_level);
         var out = await ses.run({"x": x, "tta_level": tta_level});
        """
        x = torch.rand([2, 3, 256, 256], dtype=torch.float32)
        tta_level = 2
        model = torch.jit.script(self.to_inference_model())
        torch.onnx.export(
            model,
            [x, tta_level],
            f,
            input_names=["x", "tta_level"],
            output_names=["y"],
            dynamic_axes={"x": {0: "input_batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXCreateSeamBlendingFilter(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, scale: int, offset: int, tile_size: int):
        out_channels = 3
        blend_size = 16  # FIXME: Allow variable
        model_output_size = tile_size * scale - offset * 2
        inner_tile_size = model_output_size - blend_size * 2
        x = torch.ones((out_channels, inner_tile_size, inner_tile_size), dtype=torch.float32)
        for i in range(blend_size):
            value = 1 - (1 / (blend_size + 1)) * (i + 1)
            x = F.pad(x, (1, 1, 1, 1), mode="constant", value=value)

        return x

    def export_onnx(self, f, **kwargs):
        scale = 2
        offset = 16
        tile_size = 64
        model = torch.jit.script(self.to_inference_model())
        torch.onnx.export(
            model,
            [scale, offset, tile_size],
            f,
            input_names=["scale", "offset", "tile_size"],
            output_names=["y"],
            dynamic_axes={"y": {0: "channels", 1: "height", 2: "width"}},
            **kwargs
        )


class ONNXAlphaBorderPadding(nn.Module):
    # original code at nunif/utils/alpha.py
    # make it work on onnx
    def __init__(self):
        super().__init__()
        self.sum_alpha = ChannelWiseSum(1, 3)
        self.sum_rgb = ChannelWiseSum(3, 3)
        self.eval()

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor, offset: int):
        # rgb: CHW, alpha: CHW
        rgb = rgb.clone()
        alpha = alpha.squeeze(0)
        mask = alpha.new_zeros(alpha.shape)
        mask[alpha > 0.] = 1.
        mask_nega = (mask - 1.).abs_().unsqueeze(0).expand(rgb.shape)
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
            mask = (mask_weight > 0.).float()
            mask_nega = (mask - 1.).abs_().unsqueeze(0).expand(rgb.shape)

        return rgb.clamp_(0., 1.)

    def to_inference_model(self):
        net = copy.deepcopy(self)
        net.eval()
        return net

    def to_script_module(self):
        net = self.to_inference_model()
        return torch.jit.script(net)

    def export_onnx(self, f, **kwargs):
        rgb = torch.zeros([3, 256, 256], dtype=torch.float32)
        alpha = torch.zeros([1, 256, 256], dtype=torch.float32)
        offset = torch.tensor(16, dtype=torch.int64)
        model = self.to_script_module()
        torch.onnx.export(
            model,
            [rgb, alpha, offset],
            f,
            input_names=["rgb", "alpha", "offset"],
            output_names=["y"],
            dynamic_axes={"rgb": {1: "input_height", 2: "input_width"},
                          "alpha": {1: "input_height", 2: "input_width"},
                          "y": {1: "height", 2: "width"}},
            **kwargs
        )


class ONNXScale1x(I2IBaseModel):
    # identity module for alpha channel in denoise
    def __init__(self, offset):
        super().__init__({}, scale=1, offset=offset, in_channels=3)

    def forward(self, x: torch.Tensor):
        pad = -self.i2i_offset
        return F.pad(x, (pad, pad, pad, pad), mode="constant")

    def export_onnx(self, f, **kwargs):
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        model = self.to_inference_model()
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXAntialias(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(H * 2, W * 2), mode="bilinear", align_corners=False, antialias=False)
        x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False, antialias=False)
        return x

    def export_onnx(self, f, **kwargs):
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        model = self.to_inference_model()
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch_size", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 2: "height", 3: "width"}},
            **kwargs
        )
        patch_resize_antialias(f, index=1)


class ONNXResizeBicubic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale_factor: float):
        x = F.interpolate(x, scale_factor=scale_factor,
                          mode="bicubic", align_corners=False, antialias=False)
        return x

    def export_onnx(self, f, **kwargs):
        kwargs["opset_version"] = 18
        x = torch.rand([1, 3, 256, 256], dtype=torch.float32)
        model = torch.jit.script(self.eval())
        scale_factor = float(0.75)
        torch.onnx.export(
            model,
            [x, scale_factor],
            f,
            input_names=["x", "scale_factor"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch_size", 1: "channels", 2: "input_height", 3: "input_width"},
                          "y": {0: "batch_size", 1: "channels", 2: "height", 3: "width"}},
            **kwargs
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


def _test_resize():
    import onnx
    resize = ONNXResizeBicubic()
    resize.export_onnx("./tmp/resize_bicubic.onnx")
    model = onnx.load("./tmp/resize_bicubic.onnx")
    print(model.graph)


def _test_antialias():
    import onnx
    resize = ONNXAntialias()
    resize.export_onnx("./tmp/antialias.onnx")
    model = onnx.load("./tmp/antialias.onnx")
    print(model.graph)


def _test_pad():
    import onnx
    pad = ONNXReflectionPadding()
    pad.export_onnx("./tmp/pad_reflect.onnx")
    model = onnx.load("./tmp/pad_reflect.onnx")
    print(model.graph)

    pad = ONNXReplicationPadding()
    pad.export_onnx("./tmp/pad_replicate.onnx")
    model = onnx.load("./tmp/pad_replicate.onnx")
    print(model.graph)


def _test_tta():
    import onnx
    tta_split = ONNXTTASplit()
    tta_split.export_onnx("./tmp/tta_split.onnx")
    model = onnx.load("./tmp/tta_split.onnx")
    print(model.graph)

    tta_merge = ONNXTTAMerge()
    tta_merge.export_onnx("./tmp/tta_merge.onnx")
    model = onnx.load("./tmp/tta_merge.onnx")
    print(model.graph)


def _test_blend_filter():
    import onnx
    pad = ONNXCreateSeamBlendingFilter()
    pad.export_onnx("./tmp/create_seam_blending_filter.onnx")
    model = onnx.load("./tmp/create_seam_blending_filter.onnx")
    print(model.graph)


def _test_alpha_border():
    import onnxruntime as ort
    import numpy as np
    import cv2
    from ..utils.alpha import AlphaBorderPadding
    from ..utils import pil_io

    pad = ONNXAlphaBorderPadding()
    pad.export_onnx("./tmp/alpha_border_padding.onnx")

    ses = ort.InferenceSession("./tmp/alpha_border_padding.onnx",
                               providers=["CUDAExecutionProvider"])
    for i in ses.get_inputs():
        print(i.name, i.shape, i.type)
    for i in ses.get_outputs():
        print(i.name, i.shape, i.type)

    im = cv2.imread("./tmp/alpha2.png", cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    rgb = im[:, :, 0:3].transpose(2, 0, 1).astype(np.float32) / 255.0
    alpha = im[:, :, 3:4].transpose(2, 0, 1).astype(np.float32) / 255.0
    offset = np.array([16], dtype=np.int64)
    y = ses.run(["y"], {"rgb": rgb, "alpha": alpha, "offset": offset})[0]

    y = np.clip(y * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    print(y.shape)

    pil_io.cv2_to_pil(y).show()

    im, _ = pil_io.load_image("./tmp/alpha2.png", keep_alpha=True)
    pad = AlphaBorderPadding()
    t, alpha = pil_io.to_tensor(im, return_alpha=True)
    with torch.inference_mode():
        y = pad(t, alpha, offset=16)
    pil_io.to_image(y).show()


if __name__ == "__main__":
    # _test_pad()
    # _test_tta()
    # _test_blend_filter()
    # _test_alpha_border()
    # _test_resize()
    # _test_antialias()
    pass
