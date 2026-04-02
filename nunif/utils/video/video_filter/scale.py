import torch
import torch.nn.functional as F
from .utils import get_evaluator
from typing import Any


class ScaleFilter:
    w_expr: str
    h_expr: str
    evaluator: Any

    def __init__(self, options: str):
        self.w_expr = "iw"
        self.h_expr = "ih"

        # Split options by ':' and parse key=value or positional arguments.
        # Format: "width:height"
        parts = options.split(":")
        for i, part in enumerate(parts):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                if k in ("w", "width"):
                    self.w_expr = v
                elif k in ("h", "height"):
                    self.h_expr = v
            else:
                # Positional arguments: w:h
                if i == 0:
                    self.w_expr = part
                elif i == 1:
                    self.h_expr = part

        self.evaluator = get_evaluator()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input width and height from tensor (..., H, W)
        ih, iw = x.shape[-2:]

        # Variables for expression evaluation
        # Subsampling (hsub, vsub) is fixed to 1 for RGB.
        self.evaluator.names.update(
            {
                "iw": iw,
                "in_w": iw,
                "ih": ih,
                "in_h": ih,
                "a": iw / ih,
                "sar": 1.0,
                "dar": iw / ih,
                "hsub": 1,
                "vsub": 1,
            }
        )

        # Evaluate expressions
        try:
            ow = int(self.evaluator.eval(self.w_expr))
            oh = int(self.evaluator.eval(self.h_expr))
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate scale expressions '{self.w_expr}:{self.h_expr}': {e}"
            ) from e

        # Handle aspect ratio preservation (-1, -n)
        aspect = iw / ih
        if ow < 0 and oh < 0:
            ow, oh = iw, ih
        elif ow < 0:
            n_scale = abs(ow)
            ow = int((oh * aspect) / n_scale + 0.5) * n_scale
        elif oh < 0:
            n_scale = abs(oh)
            oh = int((ow / aspect) / n_scale + 0.5) * n_scale

        # F.interpolate expects 4D input (B, C, H, W)
        input_4d = x.unsqueeze(0) if x.dim() == 3 else x
        output = F.interpolate(
            input_4d,
            size=(oh, ow),
            mode="bilinear",
            align_corners=False,
        )

        return output.squeeze(0) if x.dim() == 3 else output


def _test() -> None:
    import torch

    def test_scale(expr: str, w: int, h: int) -> None:
        print(f"Testing '{expr}' with input {w}x{h}...")
        tensor = torch.zeros((3, h, w))
        s = ScaleFilter(expr)
        output = s(tensor)
        print(f"Result: {output.shape[-1]}x{output.shape[-2]}")

    print("--- Start ScaleFilter tests (Always Bilinear) ---")
    # Case 1: Simple scale
    test_scale("1280:720", 1920, 1080)
    # Case 2: Maintain aspect ratio (-1)
    test_scale("-1:540", 1920, 1080)
    # Case 3: Multiple of 2 (-2)
    test_scale("-2:541", 1920, 1080)
    # Case 4: Multiple of 2 (-2:1920)
    test_scale("-2:1920", 3840, 2161)
    print("--- End ScaleFilter tests ---")


if __name__ == "__main__":
    _test()
