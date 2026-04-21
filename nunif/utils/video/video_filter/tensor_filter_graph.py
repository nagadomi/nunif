import re
from typing import Any, Dict, List, Tuple, Type

import torch

from ..color_transform import TensorFrame
from .bob import BobFilter
from .crop import CropFilter
from .lut3d import LUT3DFilter
from .scale import ScaleFilter
from .setparams import SetParamsFilter
from .transpose import TransposeFilter


class TensorFilterGraph:
    FILTER_MAP: Dict[str, Type[Any]] = {
        "scale": ScaleFilter,
        "crop": CropFilter,
        "bob": BobFilter,
        "transpose": TransposeFilter,
        "lut3d": LUT3DFilter,
        "setparams": SetParamsFilter,
    }
    filters: List[Any]

    def __init__(self, vf: str, deny_filters: List[str] | None = None):
        self.filters = []
        deny_filters = deny_filters or []
        video_filters = self.parse_vf_option(vf)

        for name, option in video_filters:
            if name in deny_filters:
                continue
            if name in self.FILTER_MAP:
                filter_class = self.FILTER_MAP[name]
                self.filters.append(filter_class(option))
            else:
                raise ValueError(
                    f"Unsupported tensor filter '{name}'. "
                    f"Currently supported filters: {', '.join(self.FILTER_MAP.keys())}. "
                    "If you need more filters, please request implementation."
                )

    def update(self, frame: TensorFrame | None) -> TensorFrame | None:
        if frame is None:
            return None

        # Execute filter chain
        for f in self.filters:
            frame = f(frame)
        return frame

    def flush(self) -> List[torch.Tensor]:
        # Current tensor filters are all stateless (1-in, 1-out).
        return []

    @staticmethod
    def parse_vf_option(vf: str) -> List[Tuple[str, str]]:
        video_filters: List[Tuple[str, str]] = []
        vf = vf.strip()
        if not vf:
            return video_filters

        # split by ',' not preceded by '\'
        for line in re.split(r"(?<!\\),", vf):
            line = line.strip().replace(r"\,", ",")
            if line:
                # split by '=' not preceded by '\'
                col = re.split(r"(?<!\\)=", line, 1)
                if len(col) == 2:
                    filter_name, filter_option = col
                else:
                    filter_name, filter_option = col[0], ""

                filter_name = filter_name.strip().replace(r"\=", "=")
                filter_option = filter_option.strip().replace(r"\=", "=")
                video_filters.append((filter_name, filter_option))
        return video_filters


def _test() -> None:
    import os

    import torch

    from ..color_transform import ColorRange

    print("--- Start TensorFilterGraph tests ---")
    # Test complex filter chain
    if os.path.exists("color_lut/pq2bt709.cube"):
        cube_file = "color_lut/pq2bt709.cube"
    else:
        cube_file = "nunif/utils/color_lut/pq2bt709.cube"

    vf = f"scale=1280:720,crop=256:256,transpose=1,lut3d={cube_file},setparams=colorspace=bt709:color_trc=bt709"
    graph = TensorFilterGraph(vf)

    x = torch.zeros((1, 3, 1080, 1920))
    frame = TensorFrame(
        planes=x,
        pts=0,
        dts=0,
        time_base=None,
        colorspace=9,
        color_primaries=1,
        color_trc=9,
        color_range=ColorRange.MPEG,
        side_data=None,
        use_16bit=False,
    )
    print(f"Before: CS={int(frame.colorspace)}, TRC={int(frame.color_trc)}")
    # Test complex filter chain
    print(f"Filter chain: {vf}")
    output_frame = graph.update(frame)
    if output_frame is not None:
        output = output_frame.planes
        print(f"Input: {x.shape[-1]}x{x.shape[-2]}")
        print(f"Output: {output.shape[-1]}x{output.shape[-2]}")
        print(f"After:  CS={int(output_frame.colorspace)}, TRC={int(output_frame.color_trc)}")
        assert output.shape[-1] == 256
        assert output.shape[-2] == 256
        assert int(output_frame.colorspace) == 1
        assert int(output_frame.color_trc) == 1

    # Test conditional scale with escaped commas
    max_h = 1080
    target_h = 1080
    vf_scale = f"scale=if(gt(ih\\,{max_h})\\,-2\\,iw):if(gt(ih\\,{max_h})\\,{target_h}\\,ih):flags=bicubic"
    print(f"\nTesting conditional scale: {vf_scale}")
    graph_scale = TensorFilterGraph(vf_scale)
    # Reset frame for fresh test
    frame.planes = torch.zeros((1, 3, 2160, 3840)) # 4K input
    output_frame_scale = graph_scale.update(frame)
    if output_frame_scale is not None:
        print(f"Input size: {frame.planes.shape[-1]}x{frame.planes.shape[-2]}")
        print(f"Output size: {output_frame_scale.planes.shape[-1]}x{output_frame_scale.planes.shape[-2]}")
        # 2160 > 1080, so it should scale to 1080 height
        assert output_frame_scale.planes.shape[-2] == 1080
        assert output_frame_scale.planes.shape[-1] == 1920 # Aspect ratio preserved

    # Test conditional scale skip
    vf_scale_skip = "scale=if(gt(ih\\,3000)\\,1280\\,iw):ih"
    print(f"\nTesting conditional scale skip: {vf_scale_skip}")
    graph_skip = TensorFilterGraph(vf_scale_skip)
    frame.planes = torch.zeros((1, 3, 1080, 1920))
    output_skip = graph_skip.update(frame)
    if output_skip is not None:
        print(f"Input size: {frame.planes.shape[-1]}x{frame.planes.shape[-2]}")
        print(f"Output size: {output_skip.planes.shape[-1]}x{output_skip.planes.shape[-2]}")
        # 1080 < 3000, so it should skip (remain 1920x1080)
        assert output_skip.planes.shape[-1] == 1920
        assert output_skip.planes.shape[-2] == 1080

    # Test unsupported filter
    try:
        TensorFilterGraph("unsupported_filter=test")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("--- End TensorFilterGraph tests ---")


if __name__ == "__main__":
    _test()
