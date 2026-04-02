import torch
from .utils import get_evaluator
from typing import Any


class CropFilter:
    w_expr: str
    h_expr: str
    x_expr: str
    y_expr: str
    evaluator: Any

    def __init__(self, options: str):
        self.w_expr = "iw"
        self.h_expr = "ih"
        self.x_expr = "(iw-ow)/2"
        self.y_expr = "(ih-oh)/2"

        # Split options by ':' and parse key=value or positional arguments.
        # Format: "w:h:x:y" or "w=...:h=...:x=...:y=..."
        parts = options.split(":")
        for i, part in enumerate(parts):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                if k in ("w", "width"):
                    self.w_expr = v
                elif k in ("h", "height"):
                    self.h_expr = v
                elif k == "x":
                    self.x_expr = v
                elif k == "y":
                    self.y_expr = v
            else:
                # Positional arguments: w:h:x:y
                if i == 0:
                    self.w_expr = part
                elif i == 1:
                    self.h_expr = part
                elif i == 2:
                    self.x_expr = part
                elif i == 3:
                    self.y_expr = part

        self.evaluator = get_evaluator()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input width and height from tensor (..., H, W)
        ih, iw = x.shape[-2:]

        # Initial context for evaluating w and h
        names = {
            "iw": iw,
            "in_w": iw,
            "ih": ih,
            "in_h": ih,
            "a": iw / ih,
            "sar": 1.0,
            "dar": iw / ih,
        }
        self.evaluator.names.update(names)

        # Evaluate width and height first
        try:
            ow = int(self.evaluator.eval(self.w_expr))
            oh = int(self.evaluator.eval(self.h_expr))
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate crop w:h expressions '{self.w_expr}:{self.h_expr}': {e}"
            ) from e

        # Update context for evaluating y and x (FFmpeg evaluates y before x)
        self.evaluator.names.update({"ow": ow, "out_w": ow, "oh": oh, "out_h": oh})

        try:
            # FFmpeg evaluates y then x, allowing x to depend on y.
            oy = int(self.evaluator.eval(self.y_expr))
            self.evaluator.names.update({"y": oy})
            ox = int(self.evaluator.eval(self.x_expr))
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate crop x:y expressions '{self.x_expr}:{self.y_expr}': {e}"
            ) from e

        # Clamp results to image boundary
        ox = max(0, min(ox, iw - ow))
        oy = max(0, min(oy, ih - oh))

        # Perform crop on the tensor
        return x[..., oy : oy + oh, ox : ox + ow]


def _test() -> None:
    import torch

    def test_crop(expr: str, w: int, h: int) -> None:
        print(f"Testing '{expr}' with input {w}x{h}...")
        tensor = torch.zeros((3, h, w))
        c = CropFilter(expr)
        output = c(tensor)
        print(f"Result: {output.shape[-1]}x{output.shape[-2]}")

    print("--- Start CropFilter tests ---")
    # Case 1: Fixed size (centered)
    test_crop("256:256", 1920, 1080)
    # Case 2: x=1:y=1:w=256:h=256
    test_crop("x=1:y=1:w=256:h=256", 1920, 1080)
    # Case 3: PHI (1000 : 1000/1.618... = 618)
    test_crop("in_w:1/PHI*in_w", 1000, 1000)
    # Case 4: x depends on y (y=ih/4=270, x=y=270)
    test_crop("iw/2:ih/2:x=y:y=ih/4", 1920, 1080)
    print("--- End CropFilter tests ---")


if __name__ == "__main__":
    _test()
