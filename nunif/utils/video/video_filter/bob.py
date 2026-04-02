import torch
import torch.nn.functional as F


class BobFilter:
    type: str

    def __init__(self, options: str = ""):
        self.type = "top"
        if options:
            # Parse options: "type=top" or just "top"
            parts = options.split(":")
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    if k == "type":
                        self.type = v
                else:
                    # Positional argument
                    self.type = part

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input width and height from tensor (..., H, W)
        ih, iw = x.shape[-2:]

        # Extract field (height becomes ih/2)
        field: torch.Tensor
        if self.type == "top":
            # Even lines: 0, 2, 4...
            field = x[..., 0::2, :]
        elif self.type == "bottom":
            # Odd lines: 1, 3, 5...
            field = x[..., 1::2, :]
        else:
            raise ValueError(f"Invalid field type: {self.type}")

        # Scale field back to original height (ih/2 -> ih)
        # F.interpolate expects 4D input
        input_4d = field.unsqueeze(0) if field.dim() == 3 else field
        output = F.interpolate(
            input_4d,
            size=(ih, iw),
            mode="bilinear",
            align_corners=False,
        )

        return output.squeeze(0) if x.dim() == 3 else output


def _test() -> None:
    import torch

    def test_bob(options: str, w: int, h: int) -> None:
        print(f"Testing Bob '{options}' with input {w}x{h}...")
        tensor = torch.zeros((3, h, w))
        b = BobFilter(options)
        output = b(tensor)
        print(f"Result: {output.shape[-1]}x{output.shape[-2]}")

    print("--- Start BobFilter tests ---")
    # Case 1: Default (top field)
    test_bob("", 1920, 1080)
    # Case 2: Bottom field
    test_bob("type=bottom", 1920, 1080)
    # Case 3: Positional argument
    test_bob("bottom", 1920, 1080)
    print("--- End BobFilter tests ---")


if __name__ == "__main__":
    _test()
