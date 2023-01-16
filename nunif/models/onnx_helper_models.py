# helper models for onnxruntime-web
import torch
from torch.nn import functional as F
from .model import I2IBaseModel
from typing import List


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


if __name__ == "__main__":
    import onnx
    pad = ONNXReflectionPadding()
    pad.export_onnx("./tmp/pad.onnx")
    model = onnx.load("./tmp/pad.onnx")
    print(model.graph)
