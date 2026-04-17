from typing import Dict

from ..color_transform import TensorFrame


class SetParamsFilter:
    """
    Set metadata parameters for the frame.
    Usage: setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv
    """

    # Mapping from FFmpeg names to integer values
    # Note: Some values are missing in PyAV Enum, so we use integers.
    COLORSPACE_MAP = {
        "bt709": 1,
        "fcc": 4,
        "bt470bg": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "bt2020nc": 9,
        "bt2020c": 10,
    }
    PRIMARIES_MAP = {
        "bt709": 1,
        "bt470m": 4,
        "bt470bg": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "film": 8,
        "bt2020": 9,
        "smpte428": 10,
        "smpte431": 11,
        "smpte432": 12,
        "jedec-p22": 22,
        "ebu3213": 22,
    }
    TRC_MAP = {
        "bt709": 1,
        "gamma22": 4,
        "gamma28": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "linear": 8,
        "log": 9,
        "log_sqrt": 10,
        "iec61966-2-4": 11,
        "bt1361e": 12,
        "iec61966-2-1": 13,
        "srgb": 13,
        "bt2020-10": 14,
        "bt2020-12": 15,
        "smpte2084": 16,
        "pq": 16,
        "smpte428": 17,
        "arib-std-b67": 18,
        "hlg": 18,
    }
    RANGE_MAP = {
        "unspecified": 0,
        "unknown": 0,
        "tv": 1,
        "mpeg": 1,
        "limited": 1,
        "pc": 2,
        "jpeg": 2,
        "full": 2,
    }

    def __init__(self, options: str):
        self.colorspace: int | None = None
        self.color_primaries: int | None = None
        self.color_trc: int | None = None
        self.color_range: int | None = None

        if options:
            parts = options.split(":")
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "colorspace":
                        self.colorspace = self._parse_val(v, self.COLORSPACE_MAP)
                    elif k == "color_primaries":
                        self.color_primaries = self._parse_val(v, self.PRIMARIES_MAP)
                    elif k == "color_trc":
                        self.color_trc = self._parse_val(v, self.TRC_MAP)
                    elif k in ("range", "color_range"):
                        self.color_range = self._parse_val(v, self.RANGE_MAP)

    def _parse_val(self, v: str, mapping: Dict[str, int]) -> int:
        if v in mapping:
            return mapping[v]
        try:
            return int(v)
        except ValueError:
            return 0  # unspecified

    def __call__(self, frame: TensorFrame) -> TensorFrame:
        if self.colorspace is not None:
            frame.colorspace = self.colorspace
        if self.color_primaries is not None:
            frame.color_primaries = self.color_primaries
        if self.color_trc is not None:
            frame.color_trc = self.color_trc
        if self.color_range is not None:
            frame.color_range = self.color_range
        return frame


def _test() -> None:
    import torch

    print("--- Start SetParamsFilter tests ---")
    opt = "colorspace=bt709:color_primaries=bt2020:color_trc=pq:range=pc"
    f = SetParamsFilter(opt)

    x = torch.zeros((1, 3, 16, 16))
    frame = TensorFrame(
        planes=x,
        pts=0,
        dts=0,
        time_base=None,
        colorspace=0,
        color_primaries=0,
        color_trc=0,
        color_range=0,
        side_data=None,
        use_16bit=False,
    )

    print(
        f"Before: CS={int(frame.colorspace)}, PRI={int(frame.color_primaries)}, "
        f"TRC={int(frame.color_trc)}, RNG={int(frame.color_range)}"
    )
    frame = f(frame)
    print(
        f"After:  CS={int(frame.colorspace)}, PRI={int(frame.color_primaries)}, "
        f"TRC={int(frame.color_trc)}, RNG={int(frame.color_range)}"
    )

    assert int(frame.colorspace) == 1
    assert int(frame.color_primaries) == 9
    assert int(frame.color_trc) == 16
    assert int(frame.color_range) == 2
    print("Test passed!")
    print("--- End SetParamsFilter tests ---")


if __name__ == "__main__":
    _test()
