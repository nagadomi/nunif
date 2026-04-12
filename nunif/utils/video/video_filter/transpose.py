import torch
from ..color_transform import TensorFrame


class TransposeFilter:
    dir: int
    passthrough: str

    def __init__(self, options: str = ""):
        self.dir = 0
        self.passthrough = "none"

        # Format: "dir=...:passthrough=..." or "dir:passthrough"
        if options:
            parts = options.split(":")
            for i, part in enumerate(parts):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    if k == "dir":
                        self.dir = self._parse_dir(v)
                    elif k == "passthrough":
                        self.passthrough = v.strip()
                else:
                    if i == 0:
                        self.dir = self._parse_dir(part)
                    elif i == 1:
                        self.passthrough = part.strip()

    def _parse_dir(self, v: str) -> int:
        v = str(v).strip()
        mapping = {"cclock_flip": 0, "clock": 1, "cclock": 2, "clock_flip": 3}
        if v in mapping:
            return mapping[v]
        try:
            # Handle numerical values (0-7), map 4-7 to 0-3 for simplicity
            # since we handle passthrough separately.
            return int(v) % 4
        except ValueError:
            return 0

    def __call__(self, frame: TensorFrame) -> TensorFrame:
        x = frame.planes
        # x shape is (..., H, W)
        h, w = x.shape[-2:]

        # Passthrough logic
        if self.passthrough == "portrait" and h >= w:
            return frame
        if self.passthrough == "landscape" and w >= h:
            return frame

        # torch.rot90: k is count of 90-degree COUNTER-CLOCKWISE rotations.
        if self.dir == 0:  # cclock_flip: 90 CCW then Vertical Flip
            x = torch.rot90(x, k=1, dims=(-2, -1))
            x = torch.flip(x, dims=(-2,))
        elif self.dir == 1:  # clock: 90 CW (or 270 CCW)
            x = torch.rot90(x, k=-1, dims=(-2, -1))
        elif self.dir == 2:  # cclock: 90 CCW
            x = torch.rot90(x, k=1, dims=(-2, -1))
        elif self.dir == 3:  # clock_flip: 90 CW then Vertical Flip
            x = torch.rot90(x, k=-1, dims=(-2, -1))
            x = torch.flip(x, dims=(-2,))

        frame.planes = x
        return frame


def _test() -> None:
    import torch
    from ..color_transform import Colorspace, ColorRange

    def test_transpose(options: str, w: int, h: int) -> None:
        print(f"Testing Transpose '{options}' with input {w}x{h}...")
        x = torch.arange(w * h).reshape(1, h, w).float()
        frame = TensorFrame(
            planes=x,
            pts=0,
            dts=0,
            time_base=None,
            colorspace=Colorspace.ITU709,
            color_primaries=1,
            color_trc=1,
            color_range=ColorRange.MPEG,
            side_data=None,
            use_16bit=False,
        )
        t = TransposeFilter(options)
        output_frame = t(frame)
        output = output_frame.planes
        print(f"Result: {output.shape[-1]}x{output.shape[-2]}")

    print("--- Start TransposeFilter tests ---")
    # Case 1: Clockwise 90
    test_transpose("dir=clock", 640, 480)
    # Case 2: Counter-Clockwise 90
    test_transpose("2", 640, 480)
    # Case 3: Passthrough (portrait input, portrait passthrough -> skip)
    test_transpose("dir=1:passthrough=portrait", 480, 640)
    # Case 4: Passthrough (landscape input, portrait passthrough -> apply)
    test_transpose("dir=1:passthrough=portrait", 640, 480)
    print("--- End TransposeFilter tests ---")


if __name__ == "__main__":
    _test()
