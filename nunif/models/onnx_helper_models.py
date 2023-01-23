# helper models for onnxruntime-web
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from .model import I2IBaseModel


class ONNXReflectionPadding(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, left: int, right: int, top: int, bottom: int):
        return F.pad(x, (left, right, top, bottom), mode="reflect")

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
            dynamic_axes={'x': {0: 'batch_size', 2: "height", 3: "width"},
                          'y': {0: 'batch_size', 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXTTASplit(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: int):
        if tta_level == 2:
            hflip = TF.hflip(x)
            x = torch.cat([x, hflip], dim=0)
        elif tta_level == 4:
            hflip = TF.hflip(x)
            vflip = TF.vflip(x)
            vhflip = TF.hflip(vflip)
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
            dynamic_axes={'x': {0: 'batch_size', 2: "height", 3: "width"},
                          'y': {0: 'batch_size', 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXTTAMerge(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, x: torch.Tensor, tta_level: int):
        if tta_level == 2:
            x = torch.clamp((x[0] + TF.hflip(x[1])).unsqueeze(0) / 2., 0., 1.)
        elif tta_level == 4:
            hflip = TF.hflip(x[1])
            vflip = TF.vflip(x[2])
            vhflip = TF.vflip(TF.hflip(x[3]))
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
            dynamic_axes={'x': {0: 'batch_size', 2: "height", 3: "width"},
                          'y': {0: 'batch_size', 2: "height", 3: "width"}},
            **kwargs
        )


class ONNXCreateSeamBlendingFilter(I2IBaseModel):
    def __init__(self):
        super().__init__({}, scale=1, offset=0, in_channels=3)

    def forward(self, scale: int, offset: int, tile_size: int):
        out_channels = 3
        blend_size = 4  # fixed
        model_output_size = tile_size * scale - offset * 2
        inner_tile_size = model_output_size - blend_size * 2
        x = torch.ones((out_channels, inner_tile_size, inner_tile_size), dtype=torch.float32)
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.8)
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.6)
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.4)
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.2)

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
            dynamic_axes={'y': {0: "channels", 1: "height", 2: "width"}},
            **kwargs
        )


def _test_pad():
    import onnx
    pad = ONNXReflectionPadding()
    pad.export_onnx("./tmp/pad.onnx")
    model = onnx.load("./tmp/pad.onnx")
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


if __name__ == "__main__":
    pass
