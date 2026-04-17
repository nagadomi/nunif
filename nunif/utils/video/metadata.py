import math
from fractions import Fraction
from typing import Any, Dict, Optional, Set, Tuple, Union

import av
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

# Colorspace constants
COLORSPACE_UNSPECIFIED: int = 2
COLORSPACE_BT2020: int = 9

# Mapping from friendly names to standard values
COLOR_CONFIG_MAP: Dict[
    str,
    Tuple[
        Union[Colorspace, int],
        Union[ColorPrimaries, int],
        Union[ColorTrc, int],
        Union[ColorRange, int],
    ],
] = {
    "bt709-tv": (
        Colorspace.ITU709,
        ColorPrimaries.BT709,
        ColorTrc.BT709,
        ColorRange.MPEG,
    ),
    "bt709-pc": (
        Colorspace.ITU709,
        ColorPrimaries.BT709,
        ColorTrc.BT709,
        ColorRange.JPEG,
    ),
    "bt601-tv": (
        Colorspace.ITU601,
        ColorPrimaries.SMPTE170M,
        ColorTrc.SMPTE170M,
        ColorRange.MPEG,
    ),
    "bt601-pc": (
        Colorspace.ITU601,
        ColorPrimaries.SMPTE170M,
        ColorTrc.SMPTE170M,
        ColorRange.JPEG,
    ),
    "bt2020-tv": (
        COLORSPACE_BT2020,
        ColorPrimaries.BT2020,
        ColorTrc.ARIB_STD_B67,
        ColorRange.MPEG,
    ),
    "bt2020-pc": (
        COLORSPACE_BT2020,
        ColorPrimaries.BT2020,
        ColorTrc.ARIB_STD_B67,
        ColorRange.JPEG,
    ),
    "bt2020-pq-tv": (
        COLORSPACE_BT2020,
        ColorPrimaries.BT2020,
        ColorTrc.SMPTE2084,
        ColorRange.MPEG,
    ),
}


def _list_hw_format() -> Set[str]:
    formats: Set[str] = set()
    for name in av.codec.codecs_available:
        try:
            codec = av.codec.Codec(name, mode="r")
        except ValueError:
            continue
        configs = codec.hardware_configs  # type: ignore
        if configs:
            for config in configs:
                formats.add(config.format.name)
    return formats


HW_PIX_FORMATS: Set[str] = _list_hw_format()


def get_rgb_pix_fmt(use_16bit: bool) -> str:
    # NOTE: I want to use `rgb48le`, but there is a bug in av==17.0.0
    return "gbrp16le" if use_16bit else "rgb24"


def convert_fps_fraction(fps: Union[Fraction, float, int, None]) -> Optional[Fraction]:
    if fps is None:
        return None
    if isinstance(fps, (float, int)):
        fps_val = float(fps)
        if fps_val == 29.97:
            return Fraction(30000, 1001)
        elif fps_val == 23.976:
            return Fraction(24000, 1001)
        elif fps_val == 59.94:
            return Fraction(60000, 1001)
        else:
            fps_frac = Fraction(fps_val)
            fps_frac = fps_frac.limit_denominator(0x7FFFFFFF)
            if fps_frac.denominator > 0x7FFFFFFF or fps_frac.numerator > 0x7FFFFFFF:
                raise ValueError(f"FPS={fps} could not be converted to Fraction={fps_frac}")
            return fps_frac
    return fps


def parse_time(s: Union[str, int, float]) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    try:
        cols = s.split(":")
        if len(cols) == 1:
            return max(float(cols[0]), 0.0)
        elif len(cols) == 2:
            m = int(cols[0], 10)
            s_val = float(cols[1])
            return max(m * 60 + s_val, 0.0)
        elif len(cols) == 3:
            h = int(cols[0], 10)
            m = int(cols[1], 10)
            s_val = float(cols[2])
            return max(h * 3600 + m * 60 + s_val, 0.0)
        else:
            raise ValueError("time must be hh:mm:ss, mm:ss or sec format")
    except ValueError:
        raise ValueError(f"time must be hh:mm:ss, mm:ss or sec format: {s}")


class VideoMetadata:
    video_path: str
    format: av.VideoFormat
    colorspace: Union[Colorspace, int]
    color_primaries: Union[ColorPrimaries, int]
    color_trc: Union[ColorTrc, int]
    color_range: Union[ColorRange, int]
    width: int
    height: int
    time_base: Optional[Fraction]
    use_16bit: bool
    stream_frames: int
    stream_duration: Optional[int]
    guessed_rate: Optional[Fraction]
    container_duration: Optional[float]

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        with av.open(video_path, mode="r", metadata_errors="ignore") as container:
            if not len(container.streams.video) > 0:
                raise ValueError("No video stream")

            stream = container.streams.video[0]
            self.format = stream.format
            self.colorspace = stream.colorspace
            self.color_primaries = stream.color_primaries
            self.color_trc = stream.color_trc
            self.color_range = stream.color_range
            self.width = stream.width
            self.height = stream.height
            self.time_base = stream.time_base
            self.use_16bit = stream.format.components[0].bits > 8
            self.stream_frames = stream.frames
            self.stream_duration = stream.duration
            self.guessed_rate = stream.guessed_rate

            if container.duration:
                self.container_duration = float(container.duration / av.time_base)
            else:
                self.container_duration = None

    def get_fps(self) -> Optional[Fraction]:
        return self.guessed_rate

    def get_duration(self, to_int: bool = True) -> float:
        duration: float | None
        if (self.stream_duration is not None and self.time_base is not None):
            duration = float(self.stream_duration * self.time_base)
        else:
            duration = self.container_duration

        if duration is None:
            return -1

        return math.ceil(duration) if to_int else duration

    def guess_duration_by_last_packet(self) -> Optional[float]:
        with av.open(self.video_path, mode="r", metadata_errors="ignore") as container:
            stream: av.VideoStream | av.AudioStream
            if len(container.streams.video) > 0:
                stream = container.streams.video[0]
            elif len(container.streams.audio) > 0:
                stream = container.streams.audio[0]
            else:
                return None

            large_pts = 10 * 24 * 3600 * av.time_base
            container.seek(large_pts, backward=True, any_frame=False)
            last_time = None
            for packet in container.demux([stream]):
                if packet.pts is not None:
                    last_time = float(packet.pts * packet.time_base)
            return last_time

    def guess_duration(self, to_int: bool = True) -> float:
        duration: float | None = self.get_duration(to_int=False)
        if duration is not None and duration < 0:
            duration = self.guess_duration_by_last_packet()

        if duration is None:
            return -1

        return math.ceil(duration) if to_int else duration

    def guess_frames(
        self,
        fps: Optional[Fraction] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        return_duration: bool = False,
    ) -> Union[int, Tuple[int, float]]:
        fps = fps or self.get_fps()
        duration = self.guess_duration(to_int=False)

        if duration < 0 or fps is None:
            return (-1, -1) if return_duration else -1

        if start_time is not None and end_time is not None:
            duration = min(end_time, duration) - start_time
        elif start_time is not None:
            duration = max(duration - start_time, 0)
        elif end_time is not None:
            duration = min(end_time, duration)

        frames = math.ceil(duration * fps)
        return (frames, duration) if return_duration else frames

    def get_frames(self) -> int:
        if self.stream_frames > 0:
            return self.stream_frames
        else:
            frames = self.guess_frames()
            assert isinstance(frames, int)
            return frames

    def guess_pix_fmt(self, stream_pix_fmt: str) -> str:
        if stream_pix_fmt in HW_PIX_FORMATS:
            if self.format.name in {"yuv420p", "yuvj420p"}:
                return "nv12"
            elif self.format.name == "yuv420p10le":
                return "p010le"
            elif self.format.name == "yuv420p16le":
                return "p016le"
            if self.format.name in {"gbrp"}:
                return "bgr0"
            else:
                if self.format.name in {"yuv444p", "yuv422p"}:
                    return self.format.name
                raise NotImplementedError(
                    f"Unsupported format conversion: format={self.format.name}, hw format={stream_pix_fmt}"
                )
        else:
            return stream_pix_fmt

    @staticmethod
    def guess_color_range_static(format_name: str, color_range: Union[ColorRange, int]) -> ColorRange:
        if color_range in {ColorRange.MPEG, ColorRange.JPEG}:
            return ColorRange(color_range)
        if any(s in format_name for s in ("yuvj", "rgb", "gbr")):
            return ColorRange.JPEG
        return ColorRange.MPEG

    def guess_color_range(self) -> ColorRange:
        return self.guess_color_range_static(self.format.name, self.color_range)

    @staticmethod
    def guess_colorspace_static(height: int, colorspace: Union[Colorspace, int]) -> Union[Colorspace, int]:
        if colorspace != COLORSPACE_UNSPECIFIED:
            return colorspace
        return Colorspace.ITU709 if height >= 720 else Colorspace.ITU601

    def guess_colorspace(self) -> Union[Colorspace, int]:
        return self.guess_colorspace_static(self.height, self.colorspace)

    @staticmethod
    def guess_color_trc_static(colorspace: Union[Colorspace, int]) -> ColorTrc:
        if colorspace == Colorspace.ITU709:
            return ColorTrc.BT709
        elif colorspace == Colorspace.ITU601:
            return ColorTrc.SMPTE170M
        elif colorspace == COLORSPACE_BT2020:
            return ColorTrc.SMPTE2084
        else:
            return ColorTrc.UNSPECIFIED

    @staticmethod
    def guess_color_primaries_static(colorspace: Union[Colorspace, int]) -> ColorPrimaries:
        if colorspace == Colorspace.ITU709:
            return ColorPrimaries.BT709
        elif colorspace == Colorspace.ITU601:
            return ColorPrimaries.SMPTE170M
        elif colorspace == COLORSPACE_BT2020:
            return ColorPrimaries.BT2020
        else:
            return ColorPrimaries.UNSPECIFIED

    @staticmethod
    def get_target_colorspace_static(
        colorspace_mode: str,
        pix_fmt: str,
        src_colorspace: Union[Colorspace, int] = COLORSPACE_UNSPECIFIED,
        src_color_primaries: Union[ColorPrimaries, int] = ColorPrimaries.UNSPECIFIED,
        src_color_trc: Union[ColorTrc, int] = ColorTrc.UNSPECIFIED,
        src_color_range: Union[ColorRange, int] = ColorRange.UNSPECIFIED,
        src_height: int = 1080,
        src_format_name: str = "yuv420p",
    ) -> Tuple[str, Union[Colorspace, int], Union[ColorPrimaries, int], Union[ColorTrc, int], Union[ColorRange, int]]:
        original_mode = colorspace_mode
        if colorspace_mode == "auto":
            colorspace_mode = "copy"

        if colorspace_mode in COLOR_CONFIG_MAP:
            colorspace, color_primaries, color_trc, color_range = COLOR_CONFIG_MAP[colorspace_mode]
        elif colorspace_mode in {"bt709", "bt601", "bt2020"}:
            rng = VideoMetadata.guess_color_range_static(src_format_name, src_color_range)
            suffix = "pc" if rng == ColorRange.JPEG else "tv"
            colorspace, color_primaries, color_trc, color_range = COLOR_CONFIG_MAP[f"{colorspace_mode}-{suffix}"]
        elif colorspace_mode == "copy":
            colorspace = src_colorspace
            color_primaries = src_color_primaries
            color_trc = src_color_trc
            color_range = src_color_range

            if original_mode == "auto":
                if colorspace == COLORSPACE_UNSPECIFIED:
                    colorspace = VideoMetadata.guess_colorspace_static(src_height, src_colorspace)
                if color_primaries == ColorPrimaries.UNSPECIFIED:
                    color_primaries = VideoMetadata.guess_color_primaries_static(colorspace)
                if color_trc == ColorTrc.UNSPECIFIED:
                    color_trc = VideoMetadata.guess_color_trc_static(colorspace)
                if color_range == ColorRange.UNSPECIFIED:
                    color_range = VideoMetadata.guess_color_range_static(src_format_name, src_color_range)
        elif colorspace_mode == "unspecified":
            colorspace = COLORSPACE_UNSPECIFIED
            color_primaries = ColorPrimaries.UNSPECIFIED
            color_trc = ColorTrc.UNSPECIFIED
            color_range = ColorRange.UNSPECIFIED
        else:
            colorspace, color_primaries, color_trc, color_range = COLOR_CONFIG_MAP["bt709-tv"]

        if color_range == ColorRange.JPEG:
            if pix_fmt == "yuv420p":
                pix_fmt = "yuvj420p"
            elif pix_fmt == "yuv444p":
                pix_fmt = "yuvj444p"

        return pix_fmt, colorspace, color_primaries, color_trc, color_range

    def get_target_colorspace(
        self, colorspace_mode: str, pix_fmt: str
    ) -> Tuple[
        str,
        Union[Colorspace, int],
        Union[ColorPrimaries, int],
        Union[ColorTrc, int],
        Union[ColorRange, int],
    ]:
        return self.get_target_colorspace_static(
            colorspace_mode,
            pix_fmt,
            src_colorspace=self.colorspace,
            src_color_primaries=self.color_primaries,
            src_color_trc=self.color_trc,
            src_color_range=self.color_range,
            src_height=self.height,
            src_format_name=self.format.name,
        )

    def get_reformat_options(
        self,
        target_colorspace: Union[Colorspace, int],
        target_color_primaries: Union[ColorPrimaries, int],
        target_color_trc: Union[ColorTrc, int],
    ) -> Dict[str, Any]:
        return {
            "src_colorspace": self.colorspace,
            "src_color_primaries": self.color_primaries,
            "src_color_trc": self.color_trc,
            "src_color_range": self.color_range,
            "dst_colorspace": target_colorspace,
            "dst_color_primaries": target_color_primaries,
            "dst_color_trc": target_color_trc,
            "dst_color_range": ColorRange.JPEG,
        }

    def get_auto_reformat_options(self) -> Dict[str, Any]:
        _, colorspace, color_primaries, color_trc, _ = self.get_target_colorspace("auto", self.format.name)
        return self.get_reformat_options(colorspace, color_primaries, color_trc)
