from av.video.reformatter import Colorspace, ColorTrc, ColorPrimaries, ColorRange
import torch
from fractions import Fraction
from typing import Any, Callable, Dict, Optional, Union



class VideoOutputConfig:
    pix_fmt: str
    fps: int | float | Fraction
    output_fps: Optional[str]
    options: Dict[str, str]
    container_options: Dict[str, str]
    output_width: Optional[int]
    output_height: Optional[int]
    colorspace: Optional[str]
    container_format: Optional[str]
    video_codec: Optional[str]
    state_updated: Optional[Callable[["VideoOutputConfig"], None]]
    state: Dict[str, Any]
    device: Optional[torch.device]

    def __init__(
        self,
        pix_fmt: str = "yuv420p",
        fps: int | float | Fraction = 30,
        options: Dict[str, str] = {},
        container_options: Dict[str, str] = {},
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        colorspace: Optional[str] = None,
        container_format: Optional[str] = None,
        video_codec: Optional[str] = None,
        output_fps: Optional[str] = None,
        device: Optional[torch.device] = None,
        output_colorspace: Optional[Union[Colorspace, int]] = None,
        output_color_primaries: Optional[Union[ColorPrimaries, int]] = None,
        output_color_trc: Optional[Union[ColorTrc, int]] = None,
        source_color_range: Optional[Union[ColorRange, int]] = None,
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
