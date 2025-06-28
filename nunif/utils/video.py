import av
from av.video.reformatter import ColorRange, Colorspace
from packaging import version as packaging_version
import os
from os import path
import math
from tqdm import tqdm
from PIL import Image
import mimetypes
import re
import torch
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import time
import numpy as np


# Add video mimetypes that does not exist in mimetypes
mimetypes.add_type("video/x-ms-asf", ".asf")
mimetypes.add_type("video/x-ms-vob", ".vob")
mimetypes.add_type("video/divx", ".divx")
mimetypes.add_type("video/3gpp", ".3gp")
mimetypes.add_type("video/ogg", ".ogg")
mimetypes.add_type("video/3gpp2", ".3g2")
mimetypes.add_type("video/m2ts", ".m2ts")
mimetypes.add_type("video/m2ts", ".m2t")
mimetypes.add_type("video/m2ts", ".mts")
mimetypes.add_type("video/m2ts", ".ts")
mimetypes.add_type("video/vnd.rn-realmedia", ".rm")  # fake
mimetypes.add_type("video/x-flv", ".flv")  # Not defined on Windows
mimetypes.add_type("video/x-matroska", ".mkv")  # May not be defined for some reason


VIDEO_EXTENSIONS = [
    ".mp4", ".m4v", ".mkv", ".mpeg", ".mpg", ".avi", ".wmv", ".mov", ".flv", ".webm",
    ".asf", ".vob", ".divx", ".3gp", ".ogg", ".3g2", ".m2ts", ".ts", ".rm",
]


def list_videos(directory, extensions=VIDEO_EXTENSIONS):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[-1].lower() in extensions
    )


# Color space values that are not defined in pyav.
# I am not sure about the compatibility of these values with libsws,
# but it seems to work.
ADDITIONAL_COLORSPACE_VALUES = {
    "UNSPECIFIED": 2,
    "SMPTE170M_2": 6,  # BT.601 NTSC. smpte170m is defined as 5 in libsws
    "SMPTE240M_2": 7,  # smpte240m is defined as 5 in libsws
    "BT2020": 9,
}
AV_VERSION_14 = packaging_version.parse(av.__version__).major >= 14
if not AV_VERSION_14:
    for name, value in ADDITIONAL_COLORSPACE_VALUES.items():
        if getattr(Colorspace, "_by_value") and getattr(Colorspace, "_create") and value not in Colorspace._by_value:
            Colorspace._create(name, value)


COLORSPACE_UNSPECIFIED = 2
COLORSPACE_SMPTE170M = 6
COLORSPACE_SMPTE240M = 7
COLORSPACE_BT2020 = 9
KNOWN_COLORSPACES = {Colorspace.ITU601.value, Colorspace.ITU709.value,
                     COLORSPACE_SMPTE170M, COLORSPACE_SMPTE240M, COLORSPACE_BT2020}


if "libx264" in av.codecs_available:
    LIBH264 = "libx264"
elif "libopenh264" in av.codecs_available:
    LIBH264 = "libopenh264"
else:
    LIBH264 = ""


def add_stream_from_template(container, template):
    # wrapper for av >= 14 compatibility
    if AV_VERSION_14:
        return container.add_stream_from_template(template)
    else:
        return container.add_stream(template=template)


def is_bt709(stream):
    return (stream.codec_context.color_primaries == 1 and
            stream.codec_context.color_trc == 1 and
            stream.codec_context.colorspace == 1)


def is_bt601(stream):
    # bt470bg/bt470bg/smpte170m
    return (stream.codec_context.color_primaries == 5 and
            stream.codec_context.color_trc == 6 and
            stream.codec_context.colorspace == 5)


def has_nvenc():
    return ("h264_nvenc" in av.codec.codecs_available and
            "hevc_nvenc" in av.codec.codecs_available)


def get_fps(stream):
    return stream.guessed_rate


def guess_frames(stream, fps=None, start_time=None, end_time=None, container_duration=None):
    fps = fps or get_fps(stream)
    duration = get_duration(stream, container_duration, to_int=False)

    if duration is None:
        # N/A
        return -1

    if start_time is not None and end_time is not None:
        duration = min(end_time, duration) - start_time
    elif start_time is not None:
        duration = max(duration - start_time, 0)
    elif end_time is not None:
        duration = min(end_time, duration)
    else:
        pass

    return math.ceil(duration * fps)


def get_duration(stream, container_duration=None, to_int=True):
    if stream.duration:
        duration = float(stream.duration * stream.time_base)
    else:
        duration = container_duration

    if duration is None:
        # N/A
        return -1

    if to_int:
        return math.ceil(duration)
    else:
        return duration


def get_frames(stream, container_duration=None):
    if stream.frames > 0:
        return stream.frames
    else:
        # frames is unknown
        return guess_frames(stream, container_duration=container_duration)


def from_image(im):
    return av.video.frame.VideoFrame.from_image(im)


def to_tensor(frame, device=None):
    x = torch.from_numpy(frame.to_ndarray(format="rgb24"))
    if device is not None:
        x = x.to(device)
    # CHW float32
    return x.permute(2, 0, 1).contiguous() / 255.0


def from_tensor(x):
    x = (x.permute(1, 2, 0).contiguous() * 255.0).to(torch.uint8).detach().cpu().numpy()
    return from_ndarray(x)


def from_ndarray(x):
    return av.video.frame.VideoFrame.from_ndarray(x, format="rgb24")


def to_frame(x):
    if torch.is_tensor(x):
        # float CHW
        return from_tensor(x)
    elif isinstance(x, np.ndarray):
        # uint8 HWC
        return from_ndarray(x)
    elif isinstance(x, av.video.frame.VideoFrame):
        return x
    else:
        return from_image(x)


def _print_len(stream):
    print("frames", stream.frames)
    print("guessed_frames", guess_frames(stream))
    print("duration", get_duration(stream))
    print("base_rate", float(stream.base_rate))
    print("average_rate", float(stream.average_rate))
    print("guessed_rate", float(stream.guessed_rate))


def convert_known_fps(fps):
    if isinstance(fps, float):
        if fps == 29.97:
            return Fraction(30000, 1001)
        elif fps == 23.976:
            return Fraction(24000, 1001)
        elif fps == 59.94:
            return Fraction(60000, 1001)
        else:
            fps_frac = Fraction(fps)
            fps_frac = fps_frac.limit_denominator(0x7fffffff)
            if fps_frac.denominator > 0x7fffffff or fps_frac.numerator > 0x7fffffff:
                raise ValueError(f"FPS={fps} could not be converted to Fraction={fps_frac}")
            return fps_frac

    return fps


class FixedFPSFilter():
    @staticmethod
    def parse_vf_option(vf):
        video_filters = []
        vf = vf.strip()
        if not vf:
            return video_filters

        for line in re.split(r'(?<!\\),', vf):
            line = line.strip()
            if line:
                col = re.split(r'(?<!\\)=', line, 1)
                if len(col) == 2:
                    filter_name, filter_option = col
                else:
                    filter_name, filter_option = col[0], ""
                filter_name, filter_option = filter_name.strip(), filter_option.strip()
                video_filters.append((filter_name, filter_option))
        return video_filters

    @staticmethod
    def build_graph(graph, template_stream, video_filters):
        buffer = graph.add_buffer(template=template_stream)
        prev_filter = buffer
        for filter_name, filter_option in video_filters:
            new_filter = graph.add(filter_name, filter_option if filter_option else None)
            prev_filter.link_to(new_filter)
            prev_filter = new_filter
        buffersink = graph.add("buffersink")
        prev_filter.link_to(buffersink)
        graph.configure()

    def __init__(self, video_stream, fps, vf="", deny_filters=[], colorspace=None):
        self.graph = av.filter.Graph()
        video_filters = self.parse_vf_option(vf)
        if colorspace is not None:
            video_filters.append(("colorspace", colorspace))
        if fps is not None:
            video_filters.append(("fps", str(fps)))
        video_filters = [(name, option) for name, option in video_filters if name not in deny_filters]
        self.build_graph(self.graph, video_stream, video_filters)

    def update(self, frame):
        self.graph.push(frame)
        try:
            return self.graph.pull()
        except av.error.BlockingIOError:
            return None
        except av.error.EOFError:
            # finished
            return None


class VideoFilter(FixedFPSFilter):
    def __init__(self, video_stream, vf):
        super().__init__(video_stream, fps=None, vf=vf)
        self.dummy = not vf

    def update(self, frame):
        if self.dummy:
            return frame
        else:
            return super().update(frame)


class VideoOutputConfig():
    def __init__(self, pix_fmt="yuv420p", fps=30, options={}, container_options={},
                 output_width=None, output_height=None, colorspace=None,
                 container_format=None,
                 video_codec=None, output_fps=None):
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
            self.colorspace = "unspecified"
        self.container_format = container_format
        self.video_codec = video_codec

        self.state_updated = lambda config: None
        self.state = dict(
            rgb24_options={},
            reformatter=lambda frame: frame,
            source_color_range=None,
            output_colorspace=None,
        )

    def __repr__(self):
        return "VideoOutputConfig({!r})".format(self.__dict__)


def get_default_video_codec(container_format):
    if container_format in {"mp4", "mkv"}:
        return LIBH264
    elif container_format == "avi":
        return "utvideo"
    else:
        raise ValueError(f"Unsupported container format: {container_format}")


def default_config_callback(stream):
    fps = get_fps(stream)
    if float(fps) > 30:
        fps = 30
    return VideoOutputConfig(
        fps=fps,
        options={"preset": "ultrafast", "crf": "20"}
    )


SIZE_SAFE_FILTERS = [
    "fps", "yadif", "bwdif", "nnedi", "w3fdif", "kerndeint",
    "hflip", "vflip",
]


def test_output_size(test_callback, video_stream, vf):
    video_filter = FixedFPSFilter(video_stream, fps=60, vf=vf, deny_filters=SIZE_SAFE_FILTERS)
    empty_image = Image.new("RGB", (video_stream.codec_context.width,
                                    video_stream.codec_context.height), (128, 128, 128))
    test_frame = av.video.frame.VideoFrame.from_image(empty_image).reformat(
        format=video_stream.pix_fmt,
        src_color_range=ColorRange.JPEG, dst_color_range=video_stream.codec_context.color_range)
    pts_step = int((1. / video_stream.time_base) / 30) or 1
    test_frame.pts = pts_step

    try_count = 0
    while True:
        while True:
            frame = video_filter.update(test_frame)
            test_frame.pts = (test_frame.pts + pts_step)
            if frame is not None:
                break
            try_count += 1
            if try_count * video_stream.codec_context.width * video_stream.codec_context.height * 3 > 2000 * 1024 * 1024:
                raise RuntimeError("Unable to estimate output size of video filter")
        output_frame = get_new_frames(test_callback(frame))
        if output_frame:
            output_frame = output_frame[0]
            break
    return output_frame.width, output_frame.height


def get_new_frames(frame_or_frames_or_none):
    if frame_or_frames_or_none is None:
        return []
    elif isinstance(frame_or_frames_or_none, (list, tuple)):
        return frame_or_frames_or_none
    else:
        return [frame_or_frames_or_none]


def parse_time(s):
    try:
        cols = s.split(":")
        if len(cols) == 1:
            return max(int(cols[0], 10), 0)
        elif len(cols) == 2:
            m = int(cols[0], 10)
            s = int(cols[1], 10)
            return max(m * 60 + s, 0)
        elif len(cols) == 3:
            h = int(cols[0], 10)
            m = int(cols[1], 10)
            s = int(cols[2], 10)
            return max(h * 3600 + m * 60 + s, 0)
        else:
            raise ValueError("time must be hh:mm:ss, mm:ss or sec format")
    except ValueError:
        raise ValueError("time must be hh:mm:ss, mm:ss or sec format")


def guess_color_range(input_stream):
    if input_stream.codec_context.color_range in {ColorRange.MPEG.value, ColorRange.JPEG.value}:
        return input_stream.codec_context.color_range
    else:
        if input_stream.pix_fmt.startswith("yuv4"):
            return ColorRange.MPEG
        elif input_stream.pix_fmt.startswith("yuvj4"):
            return ColorRange.JPEG
        elif input_stream.pix_fmt.startswith("rgb") or input_stream.pix_fmt.startswith("gbr"):
            return ColorRange.JPEG
        else:
            return None  # unknown


def guess_colorspace(input_stream):
    if input_stream.codec_context.colorspace != COLORSPACE_UNSPECIFIED:
        return input_stream.codec_context.colorspace
    else:
        # FIXME: maybe old video is BT.601
        if input_stream.height >= 720:
            return Colorspace.ITU709
        else:
            return Colorspace.ITU601


def guess_rgb24_options(input_stream, target_colorspace):
    src_color_range = guess_color_range(input_stream)
    src_colorspace = guess_colorspace(input_stream)

    if src_color_range is not None and src_colorspace is not None:
        if int(target_colorspace) == COLORSPACE_UNSPECIFIED:
            target_colorspace = Colorspace.ITU601
        return dict(
            src_color_range=src_color_range, dst_color_range=ColorRange.JPEG,
            src_colorspace=src_colorspace, dst_colorspace=target_colorspace,
        )
    else:
        return {}


def guess_target_colorspace(input_stream, colorspace_arg, pix_fmt,
                            exported_output_colorspace=None, exported_source_color_range=None):
    colorspace = color_primaries = color_trc = None

    if input_stream is not None and colorspace_arg == "auto":
        colorspace_arg = "copy"
    elif input_stream is None and colorspace_arg in {"auto", "copy"}:
        # image import (generate_video)
        # use exported setting
        if exported_output_colorspace == Colorspace.ITU709.value:
            if exported_source_color_range == ColorRange.MPEG:
                colorspace_arg = "bt709-tv"
            elif exported_source_color_range == ColorRange.JPEG:
                colorspace_arg = "bt709-pc"
            else:
                # unknown
                colorspace_arg = "bt709-tv"
        elif exported_output_colorspace == Colorspace.ITU601.value:
            if exported_source_color_range == ColorRange.MPEG:
                colorspace_arg = "bt601-tv"
            elif exported_source_color_range == ColorRange.JPEG:
                colorspace_arg = "bt601-pc"
            else:
                # unknown
                colorspace_arg = "bt601-tv"
        else:
            # unknown
            colorspace_arg = "bt709-tv"

    if colorspace_arg in {"bt709", "bt709-pc", "bt709-tv"}:
        # bt709
        color_primaries = Colorspace.ITU709
        color_trc = Colorspace.ITU709
        colorspace = Colorspace.ITU709

        if colorspace_arg == "bt709":
            color_range = guess_color_range(input_stream) if input_stream is not None else None
            if color_range is None:
                if exported_source_color_range in {ColorRange.MPEG.value, ColorRange.JPEG.value}:
                    color_range = exported_source_color_range
                else:
                    color_range = ColorRange.MPEG
        elif colorspace_arg == "bt709-tv":
            color_range = ColorRange.MPEG
        elif colorspace_arg == "bt709-pc":
            color_range = ColorRange.JPEG

    elif colorspace_arg in {"bt601", "bt601-pc", "bt601-tv"}:
        # bt470bg/bt470bg/smpte170m
        color_primaries = Colorspace.ITU601
        color_trc = 6
        colorspace = Colorspace.ITU601

        if colorspace_arg == "bt601":
            color_range = guess_color_range(input_stream) if input_stream is not None else None
            if color_range is None:
                if exported_source_color_range in {ColorRange.MPEG.value, ColorRange.JPEG.value}:
                    color_range = exported_source_color_range
                else:
                    color_range = ColorRange.MPEG
        elif colorspace_arg == "bt601-tv":
            color_range = ColorRange.MPEG
        elif colorspace_arg == "bt601-pc":
            color_range = ColorRange.JPEG

    elif colorspace_arg == "copy":
        # copy from source
        # might cause an error if the value is incompatible with h264
        color_primaries = input_stream.codec_context.color_primaries
        color_trc = input_stream.codec_context.color_trc
        colorspace = input_stream.codec_context.colorspace
        color_range = input_stream.codec_context.color_range
        if color_range == ColorRange.UNSPECIFIED.value:
            color_range = guess_color_range(input_stream) if input_stream is not None else None
            if color_range is None:
                if exported_source_color_range in {ColorRange.MPEG.value, ColorRange.JPEG.value}:
                    color_range = exported_source_color_range
                else:
                    color_range = ColorRange.UNSPECIFIED.value

    if color_range == ColorRange.JPEG.value:
        # replace for full range
        if pix_fmt == "yuv420p":
            pix_fmt = "yuvj420p"
        elif pix_fmt == "yuv444p":
            pix_fmt = "yuvj444p"

    return color_primaries, color_trc, colorspace, color_range, pix_fmt


def configure_colorspace(output_stream, input_stream, config):
    assert config.colorspace in {"unspecified", "auto", "copy",
                                 "bt709", "bt709-tv", "bt709-pc",
                                 "bt601", "bt601-tv", "bt601-pc",
                                 "bt2020", "bt2020-tv", "bt2020-pc"}
    config.state["rgb24_options"] = rgb24_options = {}
    config.state["reformatter"] = reformatter = lambda frame: frame
    exported_source_color_range = config.state["source_color_range"]
    exported_output_colorspace = config.state["output_colorspace"]
    if config.pix_fmt in {"rgb24", "gbrp"} or config.colorspace == "unspecified":
        config.state["source_color_range"] = config.state["output_colorspace"] = None
        if config.state_updated:
            config.state_updated(config)
        return

    if output_stream is not None:
        color_primaries, color_trc, colorspace, color_range, pix_fmt = guess_target_colorspace(
            input_stream, config.colorspace, config.pix_fmt,
            exported_output_colorspace, exported_source_color_range,
        )
        config.pix_fmt = pix_fmt  # replace
        output_stream.codec_context.color_primaries = color_primaries
        output_stream.codec_context.color_trc = color_trc
        output_stream.codec_context.colorspace = colorspace
        output_stream.codec_context.color_range = color_range

        if output_stream.codec_context.colorspace in KNOWN_COLORSPACES:
            if input_stream is not None:
                rgb24_options = guess_rgb24_options(
                    input_stream,
                    target_colorspace=output_stream.codec_context.colorspace)
                reformatter_src_colorspace = rgb24_options["dst_colorspace"]  # output_stream.codec_context.colorspace
                reformatter_src_color_range = rgb24_options["dst_color_range"]  # ColorRange.JPEG
            else:
                # image import (generate_video)
                if exported_output_colorspace in KNOWN_COLORSPACES:
                    reformatter_src_colorspace = exported_output_colorspace
                    reformatter_src_color_range = ColorRange.JPEG
                else:
                    # NOTE: export is executed with colorspace=unspecified or rgb24 or old version,
                    #       so the conversion may not be correct
                    reformatter_src_colorspace = output_stream.codec_context.colorspace
                    if exported_source_color_range in {ColorRange.JPEG.value, ColorRange.MPEG.value}:
                        reformatter_src_color_range = exported_source_color_range
                    else:
                        reformatter_src_color_range = output_stream.codec_context.color_range

            reformatter = lambda frame: frame.reformat(
                format=config.pix_fmt,
                src_colorspace=reformatter_src_colorspace,
                dst_colorspace=output_stream.codec_context.colorspace,
                src_color_range=reformatter_src_color_range,
                dst_color_range=output_stream.codec_context.color_range)
        elif output_stream.codec_context.color_range in {ColorRange.MPEG.value, ColorRange.JPEG.value}:
            # colorspace is unspecified, use guessed value

            if input_stream is not None:
                target_colorspace = guess_colorspace(input_stream)
                rgb24_options = guess_rgb24_options(input_stream, target_colorspace=target_colorspace)
                reformatter_src_colorspace = rgb24_options["dst_colorspace"]  # output_stream.codec_context.colorspace
                reformatter_src_color_range = rgb24_options["dst_color_range"]  # ColorRange.JPEG
            else:
                # image import (generate_video)
                if exported_output_colorspace in KNOWN_COLORSPACES:
                    target_colorspace = exported_output_colorspace
                    reformatter_src_colorspace = exported_output_colorspace
                    reformatter_src_color_range = ColorRange.JPEG
                else:
                    # No guess, use ITU709
                    target_colorspace = Colorspace.ITU709
                    reformatter_src_colorspace = Colorspace.ITU709
                    reformatter_src_color_range = output_stream.codec_context.color_range

            reformatter = lambda frame: frame.reformat(
                format=config.pix_fmt,
                src_colorspace=reformatter_src_colorspace, dst_colorspace=target_colorspace,
                src_color_range=reformatter_src_color_range, dst_color_range=output_stream.codec_context.color_range)
    else:
        # hook video
        assert input_stream is not None

        if config.colorspace in {"auto", "copy"}:
            target_colorspace = guess_colorspace(input_stream)
            rgb24_options = guess_rgb24_options(input_stream, target_colorspace=target_colorspace)
        elif config.colorspace in {"bt709", "bt709-pc", "bt709-tv"}:
            rgb24_options = guess_rgb24_options(input_stream, target_colorspace=Colorspace.ITU709.value)
        elif config.colorspace in {"bt601", "bt601-pc", "bt601-tv"}:
            rgb24_options = guess_rgb24_options(input_stream, target_colorspace=Colorspace.ITU601.value)

    config.state["rgb24_options"] = rgb24_options
    config.state["reformatter"] = reformatter
    if rgb24_options:
        config.state["output_colorspace"] = int(rgb24_options["dst_colorspace"])
    if input_stream is not None:
        config.state["source_color_range"] = int(guess_color_range(input_stream))

    if config.state_updated:
        config.state_updated(config)


def configure_video_codec(config):
    if config.video_codec == "utvideo":
        if config.pix_fmt == "rgb24":
            config.pix_fmt = "gbrp"
        # override unsupported colorspace, pc is not supported
        if config.colorspace in {"bt601", "bt601-pc", "bt601-tv"}:
            config.colorspace = "bt601-tv"
        elif config.colorspace in {"bt709", "bt709-pc", "bt709-tv"}:
            config.colorspace = "bt709-tv"
        elif config.colorspace in {"auto", "copy"}:
            config.colorspace = "bt709-tv"

    if config.video_codec == "libx264":
        if config.pix_fmt in {"rgb24", "gbrp"}:
            config.video_codec = "libx264rgb"
            config.pix_fmt = "rgb24"
        else:
            if config.colorspace in {"bt2020", "bt2020-tv", "bt2020-pc"}:
                # TODO: change pix_fmt
                config.video_codec = "libx265"

    if config.video_codec in {"libx265", "h264_nvenc", "hevc_nvenc"}:
        if config.pix_fmt == "rgb24":
            config.pix_fmt = "gbrp"


def try_replace(output_path_tmp, output_path):
    try_count = 4
    while try_count >= 0:
        try:
            os.replace(output_path_tmp, output_path)
            break
        except: # noqa
            time.sleep(2)
            try_count -= 1
            if try_count <= 0:
                raise


def process_video(input_path, output_path,
                  frame_callback,
                  config_callback=default_config_callback,
                  title=None,
                  vf="",
                  stop_event=None, suspend_event=None, tqdm_fn=None,
                  start_time=None, end_time=None,
                  test_callback=None):
    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")

    output_path_tmp = path.join(path.dirname(output_path), "_tmp_" + path.basename(output_path))
    input_container = av.open(input_path)

    if input_container.duration:
        container_duration = float(input_container.duration / av.time_base)
    else:
        container_duration = None

    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    if start_time is not None:
        input_container.seek(start_time * av.time_base, backward=True, any_frame=False)

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    # _print_len(video_input_stream)
    audio_input_stream = audio_output_stream = None
    if len(input_container.streams.audio) > 0:
        # has audio stream
        audio_input_stream = input_container.streams.audio[0]

    config = config_callback(video_input_stream)
    config.fps = convert_known_fps(config.fps)
    config.output_fps = convert_known_fps(config.output_fps)

    if not config.container_format:
        config.container_format = path.splitext(output_path)[-1].lower()[1:]
    if not config.video_codec:
        config.video_codec = get_default_video_codec(config.container_format)
    configure_video_codec(config)

    output_container = av.open(output_path_tmp, 'w', options=config.container_options)
    fps_filter = FixedFPSFilter(video_input_stream, fps=config.fps, vf=vf)
    if config.output_width is not None and config.output_height is not None:
        output_size = config.output_width, config.output_height
    else:
        if test_callback is None:
            # TODO: warning
            test_callback = frame_callback
        output_size = test_output_size(test_callback, video_input_stream, vf)

    output_fps = config.output_fps or config.fps
    video_output_stream = output_container.add_stream(config.video_codec, output_fps)
    configure_colorspace(video_output_stream, video_input_stream, config)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.width = output_size[0]
    video_output_stream.height = output_size[1]
    video_output_stream.options = config.options
    rgb24_options = config.state["rgb24_options"]
    reformatter = config.state["reformatter"]

    # utvideo + flac crashes on windows media player
    # default_acodec = "flac" if config.container_format == "avi" else "aac"
    default_acodec = "aac"
    if audio_input_stream is not None:
        if audio_input_stream.rate < 16000:
            audio_output_stream = output_container.add_stream(default_acodec, 16000)
            audio_copy = False
        elif start_time is not None:
            audio_output_stream = output_container.add_stream(default_acodec, audio_input_stream.rate)
            audio_copy = False
        else:
            try:
                audio_output_stream = add_stream_from_template(output_container, template=audio_input_stream)
                audio_copy = True
            except ValueError:
                audio_output_stream = output_container.add_stream(default_acodec, audio_input_stream.rate)
                audio_copy = False

    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    # TODO: `total` may be less when start_time is specified
    total = guess_frames(video_input_stream, output_fps, start_time=start_time, end_time=end_time,
                         container_duration=container_duration)
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]

    for packet in input_container.demux(streams):
        if packet.pts is not None:
            if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                break
        if packet.stream.type == "video":
            for frame in packet.decode():
                frame = fps_filter.update(frame)
                if frame is not None:
                    frame = frame.reformat(format="rgb24", **rgb24_options) if rgb24_options else frame
                    for new_frame in get_new_frames(frame_callback(frame)):
                        new_frame = reformatter(new_frame)
                        enc_packet = video_output_stream.encode(new_frame)
                        if enc_packet:
                            output_container.mux(enc_packet)
                        pbar.update(1)
        elif packet.stream.type == "audio":
            if packet.dts is not None:
                if audio_copy:
                    packet.stream = audio_output_stream
                    output_container.mux(packet)
                else:
                    for frame in packet.decode():
                        frame.pts = None
                        enc_packet = audio_output_stream.encode(frame)
                        if enc_packet:
                            output_container.mux(enc_packet)
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    while True:
        frame = fps_filter.update(None)
        if frame is not None:
            frame = frame.reformat(format="rgb24", **rgb24_options) if rgb24_options else frame
            for new_frame in get_new_frames(frame_callback(frame)):
                new_frame = reformatter(new_frame)
                enc_packet = video_output_stream.encode(new_frame)
                if enc_packet:
                    output_container.mux(enc_packet)
                pbar.update(1)
        else:
            break

    for new_frame in get_new_frames(frame_callback(None)):
        new_frame = reformatter(new_frame)
        enc_packet = video_output_stream.encode(new_frame)
        if enc_packet:
            output_container.mux(enc_packet)
        pbar.update(1)

    packet = video_output_stream.encode(None)
    if packet:
        output_container.mux(packet)
    pbar.close()
    output_container.close()
    input_container.close()

    if not (stop_event is not None and stop_event.is_set()):
        # success
        if path.exists(output_path_tmp):
            try_replace(output_path_tmp, output_path)


def generate_video(output_path,
                   frame_generator,
                   config,
                   audio_file=None,
                   title=None, total_frames=None,
                   stop_event=None, suspend_event=None, tqdm_fn=None):

    output_path_tmp = path.join(path.dirname(output_path), "_tmp_" + path.basename(output_path))
    output_container = av.open(output_path_tmp, 'w', options=config.container_options)
    output_size = config.output_width, config.output_height

    if not config.container_format:
        config.container_format = path.splitext(output_path)[-1].lower()[1:]
    if not config.video_codec:
        config.video_codec = get_default_video_codec(config.container_format)
    configure_video_codec(config)

    video_output_stream = output_container.add_stream(config.video_codec, convert_known_fps(config.fps))
    configure_colorspace(video_output_stream, None, config)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.width = output_size[0]
    video_output_stream.height = output_size[1]
    video_output_stream.options = config.options
    reformatter = config.state["reformatter"]

    if audio_file is not None:
        input_container = av.open(audio_file)
        if input_container.duration:
            container_duration = float(input_container.duration * av.time_base)
        else:
            container_duration = None
        if len(input_container.streams.audio) > 0:
            # has audio stream
            audio_input_stream = input_container.streams.audio[0]
            if audio_input_stream.rate < 16000:
                audio_output_stream = output_container.add_stream("aac", 16000)
                audio_copy = False
            else:
                try:
                    audio_output_stream = add_stream_from_template(output_container, template=audio_input_stream)
                    audio_copy = True
                except ValueError:
                    audio_output_stream = output_container.add_stream("aac", audio_input_stream.rate)
                    audio_copy = False

            tqdm_fn = tqdm_fn or tqdm
            desc = (title + " Audio" if title else "Audio")
            ncols = len(desc) + 60
            total = get_duration(audio_input_stream, container_duration=container_duration)
            pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)
            last_sec = 0

            for packet in input_container.demux([audio_input_stream]):
                if packet.pts is not None:
                    current_sec = int(packet.pts * packet.time_base)
                    if current_sec - last_sec > 0:
                        pbar.update(current_sec - last_sec)
                        last_sec = current_sec
                if packet.dts is not None:
                    if audio_copy:
                        packet.stream = audio_output_stream
                        output_container.mux(packet)
                    else:
                        for frame in packet.decode():
                            frame.pts = None
                            enc_packet = audio_output_stream.encode(frame)
                            if enc_packet:
                                output_container.mux(enc_packet)
                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
            pbar.close()
            try:
                for packet in audio_output_stream.encode(None):
                    output_container.mux(packet)
            except ValueError:
                pass
            input_container.close()

    if stop_event is not None and stop_event.is_set():
        output_container.close()
        return

    desc = (title + " Frames" if title else "Frames")
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=total_frames, ncols=ncols)
    for frame in frame_generator():
        if frame is None:
            break
        for new_frame in get_new_frames(frame):
            new_frame = reformatter(new_frame)
            enc_packet = video_output_stream.encode(new_frame)
            if enc_packet:
                output_container.mux(enc_packet)
            pbar.update(1)
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    packet = video_output_stream.encode(None)
    if packet:
        output_container.mux(packet)
    pbar.close()
    output_container.close()

    if not (stop_event is not None and stop_event.is_set()):
        # success
        if path.exists(output_path_tmp):
            try_replace(output_path_tmp, output_path)


def process_video_keyframes(input_path, frame_callback, min_interval_sec=4., title=None, stop_event=None, suspend_event=None):
    input_container = av.open(input_path)
    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")
    if input_container.duration:
        container_duration = float(input_container.duration * av.time_base)
    else:
        container_duration = None

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    video_input_stream.codec_context.skip_frame = "NONKEY"

    max_progress = get_duration(video_input_stream, container_duration=container_duration)
    desc = (title if title else input_path)
    ncols = len(desc) + 60
    pbar = tqdm(desc=desc, total=max_progress, ncols=ncols)
    prev_sec = 0
    for frame in input_container.decode(video_input_stream):
        current_sec = math.ceil(frame.pts * video_input_stream.time_base)
        if current_sec - prev_sec >= min_interval_sec:
            frame_callback(frame)
            pbar.update(current_sec - prev_sec)
            prev_sec = current_sec
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break
    pbar.close()
    input_container.close()


def hook_frame(input_path,
               frame_callback,
               config_callback=default_config_callback,
               title=None,
               vf="",
               stop_event=None, suspend_event=None, tqdm_fn=None,
               start_time=None, end_time=None):
    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")

    input_container = av.open(input_path)
    if input_container.duration:
        container_duration = float(input_container.duration / av.time_base)
    else:
        container_duration = None

    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    if start_time is not None:
        input_container.seek(start_time * av.time_base, backward=True, any_frame=False)

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"

    config = config_callback(video_input_stream)
    config.fps = convert_known_fps(config.fps)
    configure_colorspace(None, video_input_stream, config)
    rgb24_options = config.state["rgb24_options"]

    fps_filter = FixedFPSFilter(video_input_stream, fps=config.fps, vf=vf)

    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    total = guess_frames(video_input_stream, config.fps, start_time=start_time, end_time=end_time,
                         container_duration=container_duration)
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)

    for packet in input_container.demux([video_input_stream]):
        if packet.pts is not None:
            if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                break
        for frame in packet.decode():
            frame = fps_filter.update(frame)
            if frame is not None:
                frame = frame.reformat(format="rgb24", **rgb24_options) if rgb24_options else frame
                frame_callback(frame)
                pbar.update(1)
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    while True:
        frame = fps_filter.update(None)
        if frame is not None:
            frame = frame.reformat(format="rgb24", **rgb24_options) if rgb24_options else frame
            frame_callback(frame)
            pbar.update(1)
        else:
            break

    frame_callback(None)
    input_container.close()
    pbar.close()


def export_audio(input_path, output_path, start_time=None, end_time=None,
                 title=None, stop_event=None, suspend_event=None, tqdm_fn=None):

    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")

    input_container = av.open(input_path)
    if len(input_container.streams.audio) == 0:
        input_container.close()
        return False

    if start_time is not None:
        input_container.seek(start_time * av.time_base, backward=True, any_frame=False)

    audio_input_stream = input_container.streams.audio[0]
    output_container = av.open(output_path, "w")  # expect .m4a

    if input_container.duration:
        container_duration = float(input_container.duration * av.time_base)
    else:
        container_duration = None

    if audio_input_stream.rate < 16000:
        audio_output_stream = output_container.add_stream("aac", 16000)
        audio_copy = False
    elif start_time is not None:
        audio_output_stream = output_container.add_stream("aac", audio_input_stream.rate)
        audio_copy = False
    else:
        try:
            audio_output_stream = add_stream_from_template(output_container, template=audio_input_stream)
            audio_copy = True
        except ValueError:
            audio_output_stream = output_container.add_stream("aac", audio_input_stream.rate)
            audio_copy = False

    tqdm_fn = tqdm_fn or tqdm
    desc = title if title else input_path
    ncols = len(desc) + 60
    total = get_duration(audio_input_stream, container_duration=container_duration)
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)
    last_sec = 0
    for packet in input_container.demux([audio_input_stream]):
        if packet.pts is not None:
            if end_time is not None and end_time < packet.pts * packet.time_base:
                break
            current_sec = int(packet.pts * packet.time_base)
            if current_sec - last_sec > 0:
                pbar.update(current_sec - last_sec)
                last_sec = current_sec
        if packet.dts is not None:
            if audio_copy:
                packet.stream = audio_output_stream
                output_container.mux(packet)
            else:
                for frame in packet.decode():
                    frame.pts = None
                    enc_packet = audio_output_stream.encode(frame)
                    if enc_packet:
                        output_container.mux(enc_packet)
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break
    pbar.close()

    try:
        # TODO: Maybe this is needed only when audio_copy==False but not clear
        for packet in audio_output_stream.encode(None):
            output_container.mux(packet)
    except ValueError:
        pass

    output_container.close()
    input_container.close()

    return True


class _DummyFuture():
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def done(self):
        return True


class _DummyThreadPool():
    def __init__(self):
        pass

    def submit(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        return _DummyFuture(result)

    def shutdown(self):
        pass


class FrameCallbackPool():
    """
    thread pool callback wrapper
    """
    def __init__(self, frame_callback, batch_size, device, max_workers=1, max_batch_queue=2,
                 require_pts=False, skip_pts=-1, require_flush=False,
                 preprocess_callback=None,
                 postprocess_callback=None):
        if max_workers > 0:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool = _DummyThreadPool()
        self.require_pts = require_pts
        self.require_flush = require_flush
        self.skip_pts = skip_pts
        self.frame_callback = frame_callback
        self.preprocess_callback = preprocess_callback
        self.postprocess_callback = postprocess_callback
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_batch_queue = max_batch_queue
        self.devices = device if isinstance(device, (tuple, list)) else [device]
        self.round_robin_index = 0
        self.frame_queue = []
        self.pts_queue = []
        self.batch_queue = []
        self.pts_batch_queue = []
        self.futures = []

    def make_args(self, batch, pts_batch, flush):
        if self.require_pts and self.require_flush:
            return (batch, pts_batch, flush)
        elif self.require_pts:
            return (batch, pts_batch)
        elif self.require_flush:
            return (batch, flush)
        else:
            return (batch,)

    def get_results(self, future):
        frames = future.result()
        if self.postprocess_callback is not None:
            frames = self.postprocess_callback(frames)

        return [to_frame(frame) for frame in frames] if frames is not None else []

    def submit(self, *args):
        if self.preprocess_callback is not None:
            args = self.preprocess_callback(*args)
            future = self.thread_pool.submit(self.frame_callback, args)
        else:
            future = self.thread_pool.submit(self.frame_callback, *args)

        return future

    def __call__(self, frame):
        if False:
            # for debug
            print("\n__call__",
                  "frame_queue", len(self.frame_queue),
                  "batch_queue", len(self.batch_queue),
                  "pts_queue", len(self.pts_queue),
                  "pts_batch_queue", len(self.pts_batch_queue),
                  "futures", len(self.futures))

        if frame is None:
            return self.finish()
        if frame.pts <= self.skip_pts:
            return None

        self.pts_queue.append(frame.pts)
        frame = to_tensor(frame, device=self.devices[self.round_robin_index % len(self.devices)])

        self.frame_queue.append(frame)
        if len(self.frame_queue) == self.batch_size:
            batch = torch.stack(self.frame_queue)
            self.batch_queue.append(batch)
            self.frame_queue.clear()
            self.pts_batch_queue.append(list(self.pts_queue))
            self.pts_queue.clear()
            self.round_robin_index += 1

        if self.batch_queue:
            if len(self.futures) < self.max_workers or self.max_workers <= 0:
                batch = self.batch_queue.pop(0)
                pts_batch = self.pts_batch_queue.pop(0)
                future = self.submit(*self.make_args(batch, pts_batch, False))
                self.futures.append(future)
            if len(self.batch_queue) >= self.max_batch_queue and self.futures:
                future = self.futures.pop(0)
                return self.get_results(future)
        if self.futures:
            if self.futures[0].done():
                future = self.futures.pop(0)
                return self.get_results(future)

        return None

    def finish(self):
        if self.frame_queue:
            batch = torch.stack(self.frame_queue)
            self.batch_queue.append(batch)
            self.frame_queue.clear()
            self.pts_batch_queue.append(list(self.pts_queue))
            self.pts_queue.clear()

        frame_remains = []
        while len(self.batch_queue) > 0:
            if len(self.futures) < self.max_workers or self.max_workers <= 0:
                batch = self.batch_queue.pop(0)
                pts_batch = self.pts_batch_queue.pop(0)
                future = self.submit(*self.make_args(batch, pts_batch, False))
                self.futures.append(future)
            else:
                future = self.futures.pop(0)
                frame_remains += self.get_results(future)
        while len(self.futures) > 0:
            future = self.futures.pop(0)
            frame_remains += self.get_results(future)

        if self.require_flush:
            future = self.submit(*self.make_args(None, None, True))
            frame_remains += self.get_results(future)

        return frame_remains

    def shutdown(self):
        pool = self.thread_pool
        self.thread_pool = None
        if pool is not None:
            pool.shutdown()

    def __del__(self):
        self.shutdown()


def _test_process_video():
    from PIL import ImageOps
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output video file")
    args = parser.parse_args()

    def make_config(stream):
        fps = get_fps(stream)
        if fps > 30:
            fps = 30
        return VideoOutputConfig(
            fps=fps,
            options={"preset": "ultrafast", "crf": "20"}
        )

    def process_image(frame):
        if frame is None:
            return None
        im = frame.to_image()
        mirror = ImageOps.mirror(im)
        new_im = Image.new("RGB", (im.width * 2, im.height))
        new_im.paste(im, (0, 0))
        new_im.paste(mirror, (im.width, 0))
        new_frame = frame.from_image(new_im)
        return new_frame

    process_video(args.input, args.output, config_callback=make_config, frame_callback=process_image)


def _test_export_audio():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output audio file")
    parser.add_argument("--start-time", type=str, help="start time")
    parser.add_argument("--end-time", type=str, help="end time")
    args = parser.parse_args()

    print(export_audio(args.input, args.output, start_time=args.start_time, end_time=args.end_time))


def _test_reencode():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output video file")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p",
                        choices=["yuv420p", "yuv444p", "rgb24"],
                        help="colorspace")
    parser.add_argument("--colorspace", type=str, default="unspecified",
                        choices=["auto", "unspecified", "bt709", "bt709-pc", "bt709-tv", "bt601", "bt601-pc", "bt601-tv"],
                        help="colorspace")
    parser.add_argument("--video-codec", type=str, default=LIBH264,
                        choices=["libx264", "libopenh264", "libx265", "h264_nvenc", "hevc_nvenc"],
                        help="video codec")
    parser.add_argument("--max-workers", type=int, default=0, help="max worker threads")
    parser.add_argument("--gpu", type=int, default=0, help="0: gpu, -1: cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")

    args = parser.parse_args()
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    preset = "fast" if args.video_codec in {"h264_nvenc", "hevc_nvenc"} else "ultrafast"

    def make_config(stream):
        fps = get_fps(stream)
        if fps > 30:
            fps = 30
        return VideoOutputConfig(
            fps=fps,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            video_codec=args.video_codec,
            options={"preset": preset, "crf": "20"}
        )

    def process_image(frames):
        # width x 2
        return torch.cat([frames, frames], dim=3)

    callback = FrameCallbackPool(process_image, batch_size=args.batch_size,
                                 device=device, max_workers=args.max_workers,
                                 max_batch_queue=args.max_workers + 1)

    process_video(args.input, args.output, config_callback=make_config, frame_callback=callback)


if __name__ == "__main__":
    # _test_process_video()
    # _test_export_audio()
    _test_reencode()
