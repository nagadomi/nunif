from fractions import Fraction
from typing import Any, Callable, Dict

import torch
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc


class VideoOutputConfig:
    pix_fmt: str
    fps: int | float | Fraction
    output_fps: str | None
    options: Dict[str, str]
    container_options: Dict[str, str]
    output_width: int | None
    output_height: int | None
    colorspace: str | None
    container_format: str | None
    video_codec: str | None
    state_updated: Callable[["VideoOutputConfig"], None] | None
    state: Dict[str, Any]
    device: torch.device | None

    def __init__(
        self,
        pix_fmt: str = "yuv420p",
        fps: int | float | Fraction = 30,
        options: Dict[str, str] = {},
        container_options: Dict[str, str] = {},
        output_width: int | None = None,
        output_height: int | None = None,
        colorspace: str | None = None,
        container_format: str | None = None,
        video_codec: str | None = None,
        output_fps: str | None = None,
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
        if colorspace is not None:
            self.colorspace = colorspace
        else:
            self.colorspace = "auto"
        self.container_format = container_format
        self.video_codec = video_codec
        self.device = device

        self.state_updated = lambda config: None
        self.state = dict(
            rgb24_options={},
            reformatter=lambda frame: frame,
            source_color_range=source_color_range,
            output_colorspace=output_colorspace,
            output_color_primaries=output_color_primaries,
            output_color_trc=output_color_trc,
        )

    def __repr__(self):
        return "VideoOutputConfig({!r})".format(self.__dict__)
