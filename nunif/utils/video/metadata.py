import math
from fractions import Fraction
from typing import Any, Dict, Set, Tuple

import av
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

from .hwaccel import HW_DEVICES
from .utils import RGB_8BIT, RGB_16BIT

# Colorspace constants
COLORSPACE_UNSPECIFIED: int = 2
COLORSPACE_BT2020: int = 9

# Mapping from friendly names to standard values
COLOR_CONFIG_MAP: Dict[
    str,
    Tuple[
        Colorspace | int,
        ColorPrimaries | int,
        ColorTrc | int,
        ColorRange | int,
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


def convert_fps_fraction(fps: Fraction | float | int | None) -> Fraction | None:
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


def parse_time(s: str | int | float) -> float:
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


class MediaMetadata:
    path: str | None
    time_base: Fraction | None
    stream_frames: int
    stream_duration: int | None
    container_duration: float | None

    def __init__(
        self,
        path: str | None,
        time_base: Fraction | None,
        stream_frames: int,
        stream_duration: int | None,
        container_duration: float | None,
    ) -> None:
        self.path = path
        self.time_base = time_base
        self.stream_frames = stream_frames
        self.stream_duration = stream_duration
        self.container_duration = container_duration

    def get_duration(self) -> float:
        duration: float | None
        if self.stream_duration is not None and self.time_base is not None:
            duration = float(self.stream_duration * self.time_base)
        else:
            duration = self.container_duration

        return duration if duration is not None else 0.0


class AudioMetadata(MediaMetadata):
    @classmethod
    def from_file(cls, audio_path: str) -> "AudioMetadata":
        with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
            if not len(container.streams.audio) > 0:
                raise ValueError("No audio stream")

            stream = container.streams.audio[0]
            container_duration: float | None = None
            if container.duration:
                container_duration = float(container.duration / av.time_base)

            return cls(
                path=audio_path,
                time_base=stream.time_base,
                stream_frames=stream.frames,
                stream_duration=stream.duration,
                container_duration=container_duration,
            )


class VideoMetadata(MediaMetadata):
    format: av.VideoFormat
    colorspace: Colorspace | int
    color_primaries: ColorPrimaries | int
    color_trc: ColorTrc | int
    color_range: ColorRange | int
    width: int
    height: int
    use_16bit: bool
    guessed_rate: Fraction | None

    def __init__(
        self,
        format: av.VideoFormat,
        colorspace: Colorspace | int,
        color_primaries: ColorPrimaries | int,
        color_trc: ColorTrc | int,
        color_range: ColorRange | int,
        width: int,
        height: int,
        time_base: Fraction | None,
        use_16bit: bool,
        stream_frames: int,
        stream_duration: int | None,
        guessed_rate: Fraction | None,
        container_duration: float | None,
        video_path: str | None = None,
    ) -> None:
        super().__init__(
            path=video_path,
            time_base=time_base,
            stream_frames=stream_frames,
            stream_duration=stream_duration,
            container_duration=container_duration,
        )
        self.format = format
        self.colorspace = colorspace
        self.color_primaries = color_primaries
        self.color_trc = color_trc
        self.color_range = color_range
        self.width = width
        self.height = height
        self.use_16bit = use_16bit
        self.guessed_rate = guessed_rate

    @property
    def video_path(self) -> str | None:
        return self.path

    @video_path.setter
    def video_path(self, value: str | None) -> None:
        self.path = value

    @classmethod
    def from_file(cls, video_path: str) -> "VideoMetadata":
        with av.open(video_path, mode="r", metadata_errors="ignore") as container:
            if not len(container.streams.video) > 0:
                raise ValueError("No video stream")

            stream = container.streams.video[0]
            container_duration: float | None = None
            if container.duration:
                container_duration = float(container.duration / av.time_base)

            return cls.from_stream(stream, video_path=video_path, container_duration=container_duration)

    @classmethod
    def from_stream(
        cls,
        stream: av.video.stream.VideoStream,
        video_path: str | None = None,
        container_duration: float | None = None,
    ) -> "VideoMetadata":
        if stream.format.name in HW_DEVICES:
            raise ValueError(
                f"VideoMetadata.from_stream does not support hardware streams ({stream.format.name}). "
                "Please use a software stream for accurate metadata."
            )

        return cls(
            format=stream.format,
            colorspace=stream.colorspace,
            color_primaries=stream.color_primaries,
            color_trc=stream.color_trc,
            color_range=stream.color_range,
            width=stream.width,
            height=stream.height,
            time_base=stream.time_base,
            use_16bit=stream.format.components[0].bits > 8,
            stream_frames=stream.frames,
            stream_duration=stream.duration,
            guessed_rate=stream.guessed_rate,
            container_duration=container_duration,
            video_path=video_path,
        )

    @classmethod
    def from_config(
        cls,
        width: int,
        height: int,
        pix_fmt: str = "yuv420p",
        fps: Fraction | float | int = 30,
        colorspace: Colorspace | int = Colorspace.ITU709,
        color_primaries: ColorPrimaries | int = ColorPrimaries.BT709,
        color_trc: ColorTrc | int = ColorTrc.BT709,
        color_range: ColorRange | int = ColorRange.MPEG,
    ) -> "VideoMetadata":
        fps_frac = convert_fps_fraction(fps)
        return cls(
            format=av.VideoFormat(pix_fmt),
            colorspace=colorspace,
            color_primaries=color_primaries,
            color_trc=color_trc,
            color_range=color_range,
            width=width,
            height=height,
            time_base=1 / fps_frac if fps_frac else None,
            use_16bit=pix_fmt_requires_16bit(pix_fmt),
            stream_frames=0,
            stream_duration=None,
            guessed_rate=fps_frac,
            container_duration=None,
            video_path=None,
        )

    def get_fps(self) -> Fraction | None:
        return self.guessed_rate

    def guess_duration_by_last_packet(self) -> float | None:
        if self.video_path is None:
            raise RuntimeError("guess_duration_by_last_packet requires video_path")

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
        duration: float = self.get_duration()
        if duration <= 0:
            if self.video_path is not None:
                duration = self.guess_duration_by_last_packet() or 0.0

        if duration <= 0:
            return -1

        return math.ceil(duration) if to_int else duration

    def guess_frames(
        self,
        fps: Fraction | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        return_duration: bool = False,
    ) -> int | Tuple[int, float]:
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
            elif self.format.name in {"yuv422p", "yuvj422p"}:
                return "nv16"
            elif self.format.name == "yuv420p10le":
                return "p010le"
            elif self.format.name == "yuv420p12le":
                return "p012le"
            elif self.format.name == "yuv420p16le":
                return "p016le"
            if self.format.name in {"gbrp"}:
                return "bgr0"
            else:
                return self.format.name
        else:
            return stream_pix_fmt

    def guess_sw_dlpack_pix_fmt(self) -> str | None:
        if self.format.name in {"yuv420p", "yuvj420p"}:
            return "nv12"
        elif self.format.name == "yuv420p10le":
            return "p010le"
        return None

    def guess_rgb_pix_fmt(self) -> str:
        if self.use_16bit:
            return RGB_16BIT
        else:
            return RGB_8BIT

    @staticmethod
    def guess_color_range_static(format_name: str, color_range: ColorRange | int) -> ColorRange:
        if color_range in {ColorRange.MPEG, ColorRange.JPEG}:
            return ColorRange(color_range)
        if any(s in format_name for s in ("yuvj", "rgb", "gbr")):
            return ColorRange.JPEG
        return ColorRange.MPEG

    def guess_color_range(self) -> ColorRange:
        return self.guess_color_range_static(self.format.name, self.color_range)

    @staticmethod
    def guess_colorspace_static(height: int, colorspace: Colorspace | int) -> Colorspace | int:
        if colorspace != COLORSPACE_UNSPECIFIED:
            return colorspace
        return Colorspace.ITU709 if height >= 720 else Colorspace.ITU601

    def guess_colorspace(self) -> Colorspace | int:
        return self.guess_colorspace_static(self.height, self.colorspace)

    @staticmethod
    def guess_color_trc_static(colorspace: Colorspace | int) -> ColorTrc:
        if colorspace == Colorspace.ITU709:
            return ColorTrc.BT709
        elif colorspace == Colorspace.ITU601:
            return ColorTrc.SMPTE170M
        elif colorspace == COLORSPACE_BT2020:
            return ColorTrc.SMPTE2084
        else:
            return ColorTrc.UNSPECIFIED

    @staticmethod
    def guess_color_primaries_static(colorspace: Colorspace | int) -> ColorPrimaries:
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
        src_colorspace: Colorspace | int = COLORSPACE_UNSPECIFIED,
        src_color_primaries: ColorPrimaries | int = ColorPrimaries.UNSPECIFIED,
        src_color_trc: ColorTrc | int = ColorTrc.UNSPECIFIED,
        src_color_range: ColorRange | int = ColorRange.UNSPECIFIED,
        src_height: int = 1080,
        src_format_name: str = "yuv420p",
    ) -> Tuple[str, Colorspace | int, ColorPrimaries | int, ColorTrc | int, ColorRange | int]:
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
        Colorspace | int,
        ColorPrimaries | int,
        ColorTrc | int,
        ColorRange | int,
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

    def get_input_reformat_options(
        self,
        target_colorspace: Colorspace | int,
        target_color_primaries: ColorPrimaries | int,
        target_color_trc: ColorTrc | int,
    ) -> Dict[str, Any]:
        return {
            "src_colorspace": self.guess_colorspace(),
            "src_color_primaries": self.color_primaries,
            "src_color_trc": self.color_trc,
            "src_color_range": self.guess_color_range(),
            "dst_colorspace": target_colorspace,
            "dst_color_primaries": target_color_primaries,
            "dst_color_trc": target_color_trc,
            "dst_color_range": ColorRange.JPEG,
        }

    def get_auto_input_reformat_options(self) -> Dict[str, Any]:
        _, colorspace, color_primaries, color_trc, _ = self.get_target_colorspace("auto", self.format.name)
        return self.get_input_reformat_options(colorspace, color_primaries, color_trc)


def pix_fmt_requires_16bit(pix_fmt: str) -> bool:
    return pix_fmt in {
        "yuv420p10le",
        "p010le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "yuv444p16le",
        "gbrp16le",
        "gbrp12le",
        "gbrp10le",
        "rgb48le",
    }
