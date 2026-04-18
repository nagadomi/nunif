from fractions import Fraction
from typing import Callable, Dict

import torch
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc


class VideoOutputConfig:
    pix_fmt: str
    fps: int | float | Fraction | None
    output_fps: Fraction | None
    options: Dict[str, str]
    container_options: Dict[str, str]
    output_width: int | None
    output_height: int | None
    colorspace: str
    container_format: str | None
    video_codec: str | None
    state_updated: Callable[["VideoOutputConfig"], None] | None
    device: torch.device | None

    # State properties
    output_colorspace: int | None
    output_color_primaries: int | None
    output_color_trc: int | None
    source_color_range: int | None

    def __init__(
        self,
        pix_fmt: str = "yuv420p",
        fps: int | float | Fraction | None = 30,
        options: Dict[str, str] = {},
        container_options: Dict[str, str] = {},
        output_width: int | None = None,
        output_height: int | None = None,
        colorspace: str | None = None,
        container_format: str | None = None,
        video_codec: str | None = None,
        output_fps: Fraction | None = None,
        device: torch.device | None = None,
        output_colorspace: Colorspace | int | None = None,
        output_color_primaries: ColorPrimaries | int | None = None,
        output_color_trc: ColorTrc | int | None = None,
        source_color_range: ColorRange | int | None = None,
    ):
        self.pix_fmt = pix_fmt
        self.fps = fps
        self.output_fps = output_fps
        self.options = options
        self.container_options = container_options
        self.output_width = output_width
        self.output_height = output_height
        self.colorspace = colorspace if colorspace is not None else "auto"
        self.container_format = container_format
        self.video_codec = video_codec
        self.device = device

        self.state_updated = lambda config: None

        self.output_colorspace = int(output_colorspace) if output_colorspace is not None else None
        self.output_color_primaries = int(output_color_primaries) if output_color_primaries is not None else None
        self.output_color_trc = int(output_color_trc) if output_color_trc is not None else None
        self.source_color_range = int(source_color_range) if source_color_range is not None else None

    def __repr__(self):
        return "VideoOutputConfig({!r})".format(self.__dict__)
