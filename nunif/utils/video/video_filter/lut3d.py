import torch
from ...color_lut import load_lut, apply_lut
from ..color_transform import TensorFrame


class LUT3DFilter:
    """
    3D LUT filter for color correction.
    Usage: lut3d=path/to/lut.cube
    """

    lut: torch.Tensor

    def __init__(self, options: str):
        # options is the path to the .cube file
        if not options:
            raise ValueError("lut3d filter requires a path to a .cube file")

        # Basic parsing for options if we want to support more (like interpolation mode)
        # For now, we only support the path
        self.lut_path = options.strip()
        self.lut = load_lut(self.lut_path)

    def __call__(self, frame: TensorFrame) -> TensorFrame:
        # F.grid_sample requires float32/64.
        # lut is already float32.
        # Move lut to the same device as input x
        x = frame.planes
        if self.lut.device != x.device:
            self.lut = self.lut.to(x.device)

        # Apply LUT
        frame.planes = apply_lut(x, self.lut)
        return frame


def _test() -> None:
    import os
    from ..color_transform import Colorspace, ColorRange

    print("--- Start LUT3DFilter tests ---")
    lut_path = "color_lut/pq2bt709.cube"
    if not os.path.exists(lut_path):
        print(f"Skipping test: {lut_path} not found")
        return

    filter = LUT3DFilter(lut_path)
    x = torch.zeros((1, 3, 100, 100))
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
    output_frame = filter(frame)
    output = output_frame.planes
    print(f"Input: {x.shape}, Output: {output.shape}")
    print("--- End LUT3DFilter tests ---")


if __name__ == "__main__":
    _test()
