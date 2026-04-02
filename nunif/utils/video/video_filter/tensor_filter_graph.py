import re
from .scale import ScaleFilter
from .crop import CropFilter
from .bob import BobFilter
from .transpose import TransposeFilter
from typing import Any, Dict, List, Optional, Tuple, Type
import torch


class TensorFilterGraph:
    FILTER_MAP: Dict[str, Type[Any]] = {
        "scale": ScaleFilter,
        "crop": CropFilter,
        "bob": BobFilter,
        "transpose": TransposeFilter,
    }
    filters: List[Any]

    def __init__(self, vf: str, deny_filters: Optional[List[str]] = None):
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

    def update(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None

        # Execute filter chain
        for f in self.filters:
            x = f(x)
        return x

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
    import torch

    print("--- Start TensorFilterGraph tests ---")
    # Test complex filter chain
    vf = "scale=1280:720,crop=256:256,transpose=1"
    graph = TensorFilterGraph(vf)

    x = torch.zeros((3, 1080, 1920))
    output = graph.update(x)

    print(f"Filter chain: {vf}")
    if output is not None:
        print(f"Input: {x.shape[-1]}x{x.shape[-2]}")
        print(f"Output: {output.shape[-1]}x{output.shape[-2]}")

    # Test unsupported filter
    try:
        TensorFilterGraph("unsupported_filter=test")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("--- End TensorFilterGraph tests ---")


if __name__ == "__main__":
    _test()
