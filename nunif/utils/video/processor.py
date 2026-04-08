import av
from av.video.reformatter import Colorspace, ColorTrc, ColorRange
from av.codec.hwaccel import HWAccel
import os
import gc
from os import path
import io
import math
from tqdm import tqdm
from PIL import Image
import torch
import time
import sys
from fractions import Fraction
from .output_config import VideoOutputConfig
from .frame_callback_pool import ( # noqa
    FrameCallbackPool,
    to_tensor,
    from_ndarray,
    from_image,
    from_tensor,
    to_frame,
    to_ndarray,
    get_source_dtype,
)
from .color_transform import (
    SoftwareVideoFormat,
    InputTransform,
    configure_colorspace,
    configure_video_codec,
    COLORSPACE_BT2020,
)
from .hwaccel import get_supported_hwdevices, create_hwaccel, should_use_tensor_frame
from .video_preprocessor import VideoPreprocessor
from ..color_lut import load_lut, apply_lut, get_hdr2sdr_lut_path  # noqa
from types import GeneratorType
from typing import Optional, Union, Dict, cast


VIDEO_EXTENSIONS = [
    ".mp4", ".m4v", ".mkv", ".mpeg", ".mpg", ".avi", ".wmv", ".mov", ".flv", ".webm",
    ".asf", ".vob", ".divx", ".3gp", ".ogv", ".3g2", ".m2ts", ".ts", ".rm",
]

AV_READ_OPTIONS: Dict[str, str] = dict(mode="r", metadata_errors="ignore")


def list_videos(directory, extensions=VIDEO_EXTENSIONS):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[-1].lower() in extensions
    )


RGB_8BIT = "rgb24"
RGB_16BIT = "gbrp16le"


if "libx264" in av.codecs_available:
    LIBH264 = "libx264"
elif "libopenh264" in av.codecs_available:
    LIBH264 = "libopenh264"
else:
    LIBH264 = ""


def has_nvenc():
    return ("h264_nvenc" in av.codec.codecs_available and
            "hevc_nvenc" in av.codec.codecs_available)


def has_qsv():
    return ("h264_qsv" in av.codec.codecs_available and
            "hevc_qsv" in av.codec.codecs_available)


def get_fps(stream):
    return stream.guessed_rate


def guess_frames(stream, fps=None, start_time=None, end_time=None, container_duration=None, input_path=None,
                 return_duration=False):
    fps = fps or get_fps(stream)
    duration = guess_duration(
        stream,
        container_duration=container_duration,
        input_path=input_path,
        to_int=False)

    if duration < 0:
        # N/A
        if return_duration:
            return -1, -1
        return -1

    if start_time is not None and end_time is not None:
        duration = min(end_time, duration) - start_time
    elif start_time is not None:
        duration = max(duration - start_time, 0)
    elif end_time is not None:
        duration = min(end_time, duration)
    else:
        pass

    if return_duration:
        return math.ceil(duration * fps), duration

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


def guess_duration_by_last_packet(input_path):
    with av.open(input_path, **AV_READ_OPTIONS) as container:
        if len(container.streams.video) > 0:
            stream = container.streams.video[0]
        elif len(container.streams.audio) > 0:
            stream = container.streams.audio[0]
        else:
            return None

        # Seek to the last keyframe
        large_pts = 10 * 24 * 3600 * av.time_base
        container.seek(large_pts, backward=True, any_frame=False)
        last_time = None
        # Find the last packet time
        for packet in container.demux([stream]):
            if packet.pts is not None:
                last_time = float(packet.pts * packet.time_base)
        if last_time is not None:
            return last_time
        else:
            return None


def guess_duration(stream, container_duration=None, input_path=None, to_int=True):
    if stream.duration:
        duration = float(stream.duration * stream.time_base)
    else:
        duration = container_duration

    if duration is None and input_path is not None:
        duration = guess_duration_by_last_packet(input_path)

    if duration is None:
        return -1

    if to_int:
        return math.ceil(duration)
    else:
        return duration


def get_frames(stream, container_duration=None, input_path=None):
    if container_duration is None:
        if stream.container and stream.container.duration:
            container_duration = float(stream.container.duration / av.time_base)
    if stream.frames > 0:
        return stream.frames
    else:
        # frames is unknown
        return guess_frames(stream, container_duration=container_duration, input_path=input_path)


def convert_fps_fraction(fps):
    if isinstance(fps, (float, int)):
        fps = float(fps)
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


def pix_fmt_requires_16bit(pix_fmt):
    return pix_fmt in {
        "yuv420p10le", "p010le",
        "yuv422p10le", "yuv444p10le",
        "yuv420p12le", "yuv422p12le", "yuv444p12le",
        "yuv444p16le",
        RGB_16BIT, "gbrp12le", "gbrp10le", "rgb48le",
    }


def _print_len(stream):
    print("frames", stream.frames)
    print("guessed_frames", guess_frames(stream))
    print("duration", get_duration(stream))
    print("base_rate", float(stream.base_rate))
    print("average_rate", float(stream.average_rate))
    print("guessed_rate", float(stream.guessed_rate))


def hdr2sdr(
        lut,
        av_frame, color_trc, output_colorspace,
):
    output_colorspace = output_colorspace.split("-")[0]
    assert output_colorspace in {"bt709", "bt601"}
    assert av_frame.colorspace == COLORSPACE_BT2020
    assert color_trc in {ColorTrc.SMPTE2084, ColorTrc.ARIB_STD_B67}

    x = av_frame.to_ndarray(format=RGB_16BIT,
                            src_color_range=av_frame.color_range,
                            dst_color_range=ColorRange.JPEG)
    x = torch.from_numpy(x).to(lut.device, dtype=lut.dtype).permute(2, 0, 1) / 65535.0

    x = apply_lut(x, lut)
    x = (x.clamp(0, 1) * 65535).to(torch.uint16)
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()

    output_frame = av.video.frame.VideoFrame.from_ndarray(x, format=RGB_16BIT)
    if output_colorspace == "bt709":
        output_frame.colorspace = Colorspace.ITU709
    else:
        output_frame.colorspace = Colorspace.ITU601
    output_frame.color_range = ColorRange.JPEG
    output_frame.pts = av_frame.pts
    output_frame.dts = av_frame.dts
    output_frame.time_base = av_frame.time_base
    output_frame.opaque = av_frame.opaque
    # output_frame.side_data = av_frame.side_data

    return output_frame


def update_hdr2sdr_video_filter(vf, colorspace, color_trc, output_colorspace):
    lut_path = get_lut_path(colorspace=colorspace,
                            color_trc=color_trc,
                            output_colorspace=output_colorspace)
    if lut_path is not None:
        if "bt709" in output_colorspace:
            colorspace_filter = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
        else:
            colorspace_filter = "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=smpte170m"

        lut_filter = f"lut3d={lut_path},{colorspace_filter}"
        vf = f"{vf},{lut_filter}" if vf else lut_filter
        return vf
    else:
        return vf


def get_lut_path(colorspace, color_trc, output_colorspace):
    use_hdr2sdr = (colorspace == COLORSPACE_BT2020 and
                   color_trc in {ColorTrc.SMPTE2084, ColorTrc.ARIB_STD_B67} and
                   output_colorspace in {"bt709", "bt709-tv", "bt709-pc",
                                         "bt601", "bt601-tv", "bt601-pc"})
    if use_hdr2sdr:
        return get_hdr2sdr_lut_path({
            (ColorTrc.SMPTE2084, "bt709"): "pq2bt709",
            (ColorTrc.SMPTE2084, "bt601"): "pq2bt601",
            (ColorTrc.ARIB_STD_B67, "bt709"): "hlg2bt709",
            (ColorTrc.ARIB_STD_B67, "bt601"): "hlg2bt601",
        }[(color_trc, output_colorspace.split("-")[0])])
    else:
        return None


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


def test_output_size(test_callback, video_stream, sw_format, vf):
    video_filter = VideoPreprocessor(video_stream, sw_format, fps=convert_fps_fraction(59.94),
                                     vf=vf, deny_filters=SIZE_SAFE_FILTERS)
    empty_image = Image.new("RGB", (sw_format.width, sw_format.height), (128, 128, 128))
    test_frame = av.VideoFrame.from_image(empty_image).reformat(
        format=sw_format.format.name,
        src_color_range=ColorRange.JPEG, dst_color_range=video_stream.codec_context.color_range)
    pts_step = int((1. / video_stream.time_base) / 30) or 1
    test_frame.pts = pts_step

    try_count = 0
    while True:
        while True:
            frames = video_filter.update(test_frame)
            test_frame.pts = (test_frame.pts + pts_step)
            test_frame.time_base = video_stream.time_base

            if frames:
                break
            try_count += 1
            if try_count * video_stream.codec_context.width * video_stream.codec_context.height * 3 > 2000 * 1024 * 1024:
                raise RuntimeError("Unable to estimate output size of video filter")

        frame = frames[0]
        output_frame = get_new_frames(test_callback(frame))
        if output_frame:
            output_frame = output_frame[0]
            break
    if isinstance(output_frame, av.VideoFrame):
        return output_frame.width, output_frame.height
    elif torch.is_tensor(output_frame):
        return output_frame.shape[-1], output_frame.shape[-2]
    else:
        raise ValueError(f"Unexpectged type `{type(output_frame)}` in test_output_size()")


def get_new_frames(frames):
    if frames is None:
        return []
    elif isinstance(frames, (list, tuple)):
        return frames
    elif isinstance(frames, GeneratorType):
        return frames
    else:
        return [frames]


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


def make_temporary_file_path(output_path):
    return path.join(path.dirname(output_path), "_tmp_" + path.basename(output_path))


def make_error_file_path(output_path):
    return path.join(path.dirname(output_path), "_error_" + path.basename(output_path))


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


def test_audio_copy(input_path, output_path):
    buff = io.BytesIO()
    buff.name = path.basename(output_path)
    try:
        with (
                av.open(input_path, **AV_READ_OPTIONS) as input_container,
                av.open(buff, mode="w") as output_container,
        ):
            if len(input_container.streams.audio) > 0:
                audio_input_stream = input_container.streams.audio[0]
            else:
                return True

            audio_output_stream = output_container.add_stream_from_template(audio_input_stream)
            for packet in input_container.demux([audio_input_stream]):
                if packet.dts is not None:
                    packet.stream = audio_output_stream
                    output_container.mux(packet)
                    break
    except Exception:  # noqa
        return False
    else:
        return True


def safe_decode(packet, strict=False):
    if strict:
        return packet.decode()

    try:
        frames = packet.decode()
    except av.error.InvalidDataError:  # corrupted frame
        frames = []
        print("\n[WARN] Input video has invalid data/frames! continuing anyway...", file=sys.stderr)
    except av.error.PermissionError:  # wmv drm protection
        frames = []
        print("\n[WARN] No permission to read data/frames! continuing anyway...", file=sys.stderr)
    except av.error.PatchWelcomeError:  # pyAV unimplemented
        frames = []
        print("\n[WARN] Unknown data/frames type (pyAV Unimplemented)! continuing anyway...", file=sys.stderr)
    return frames


def process_video(
        input_path, output_path,
        frame_callback,
        config_callback=default_config_callback,
        title=None,
        vf="",
        stop_event=None, suspend_event=None, tqdm_fn=None,
        start_time=None, end_time=None,
        test_callback=None,
        device: Union[str, torch.device] = "cpu",
        inference_mode: bool = True,
        hwaccel: Optional[str] = None,
        disable_software_fallback: bool = False,
):
    with torch.inference_mode(inference_mode):
        _process_video(
            input_path, output_path,
            frame_callback,
            config_callback=config_callback,
            title=title,
            vf=vf,
            stop_event=stop_event, suspend_event=suspend_event, tqdm_fn=tqdm_fn,
            start_time=start_time, end_time=end_time,
            test_callback=test_callback,
            device=device,
            hwaccel=hwaccel,
            disable_software_fallback=disable_software_fallback,
        )


def _process_video(
        input_path, output_path,
        frame_callback,
        config_callback,
        title,
        vf,
        stop_event, suspend_event, tqdm_fn,
        start_time, end_time,
        test_callback,
        device: Union[str, torch.device],
        hwaccel: Optional[str],
        disable_software_fallback: bool,
):
    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")
    if isinstance(device, str):
        device = torch.device(device)

    sw_format = SoftwareVideoFormat(input_path)
    input_hwaccel = create_hwaccel(device=hwaccel, device_id=device.index,
                                   disable_software_fallback=disable_software_fallback)
    output_path_tmp = make_temporary_file_path(output_path)
    input_container = av.open(input_path, **AV_READ_OPTIONS, hwaccel=input_hwaccel)  # type: ignore

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
    config.fps = convert_fps_fraction(config.fps)
    config.output_fps = convert_fps_fraction(config.output_fps)

    if not config.container_format:
        config.container_format = path.splitext(output_path)[-1].lower()[1:]

    if not config.video_codec:
        config.video_codec = get_default_video_codec(config.container_format)
    configure_video_codec(config)

    output_hwaccel = None
    if config.video_codec in {"h264_nvenc", "hevc_nvenc"}:
        # It seems this isn't actually necessary.
        if device.type == "cuda":
            device_id = device.index if device.index is not None else 0
        else:
            device_id = None
        output_hwaccel = HWAccel(device_type="cuda", device=device_id, options={"primary_ctx": "1"})
    output_container = av.open(output_path_tmp, mode="w", options=config.container_options, hwaccel=output_hwaccel)

    vf = update_hdr2sdr_video_filter(
        vf,
        colorspace=video_input_stream.codec_context.colorspace,
        color_trc=video_input_stream.codec_context.color_trc,
        output_colorspace=config.colorspace
    )
    output_fps = config.output_fps or config.fps
    video_output_stream = output_container.add_stream(config.video_codec, output_fps)
    configure_colorspace(video_output_stream, sw_format, config)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.options = config.options
    rgb24_options = config.state["rgb24_options"]
    reformatter = config.state["reformatter"]

    if should_use_tensor_frame(sw_format.format.name, hwaccel, device):
        input_reformatter = lambda frame: frame
        input_transform = InputTransform(
            src_pix_fmt=sw_format.format.name,
            src_colorspace=rgb24_options["src_colorspace"],
            src_color_primaries=rgb24_options["src_color_primaries"],
            src_color_trc=rgb24_options["src_color_trc"],
            src_color_range=rgb24_options["src_color_range"],
            dst_colorspace=rgb24_options["dst_colorspace"],
            dst_color_primaries=rgb24_options["dst_color_primaries"],
            dst_color_trc=rgb24_options["dst_color_trc"],
            dst_color_range=rgb24_options["dst_color_range"],
            use_16bit=sw_format.use_16bit,
            device=device,
        )
        fps_filter = VideoPreprocessor(video_input_stream, sw_format, fps=config.fps, vf=vf,
                                       input_transform=input_transform)
    else:
        dst_pix_fmt: Optional[str] = sw_format.guess_pix_fmt(video_input_stream.pix_fmt)
        if dst_pix_fmt == video_input_stream.pix_fmt:
            dst_pix_fmt = None
        input_reformatter = lambda frame: frame.reformat(
            format=dst_pix_fmt,
            src_colorspace=rgb24_options["src_colorspace"],
            src_color_range=rgb24_options["src_color_range"],
            dst_colorspace=rgb24_options["dst_colorspace"],
            dst_color_primaries=rgb24_options["dst_color_primaries"],
            dst_color_trc=rgb24_options["dst_color_trc"],
            dst_color_range=rgb24_options["dst_color_range"],
        )
        fps_filter = VideoPreprocessor(video_input_stream, sw_format, fps=config.fps, vf=vf)

    if config.output_width is not None and config.output_height is not None:
        output_size = config.output_width, config.output_height
    else:
        if test_callback is None:
            # TODO: warning
            test_callback = frame_callback
        output_size = test_output_size(test_callback, video_input_stream, sw_format, vf)
    video_output_stream.width = output_size[0]
    video_output_stream.height = output_size[1]

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
            if test_audio_copy(input_path, output_path):
                audio_output_stream = output_container.add_stream_from_template(template=audio_input_stream)
                audio_copy = True
            else:
                audio_output_stream = output_container.add_stream(default_acodec, audio_input_stream.rate)
                audio_copy = False

    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    # TODO: `total` may be less when start_time is specified
    total = guess_frames(video_input_stream,
                         fps=output_fps,
                         start_time=start_time, end_time=end_time,
                         container_duration=container_duration,
                         input_path=input_path)
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]

    try:
        for i, packet in enumerate(input_container.demux(streams)):
            if packet.pts is not None:
                if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                    break
            if packet.stream.type == "video":
                for frame in safe_decode(packet, strict=disable_software_fallback):
                    for out_frame in fps_filter.update(frame):
                        ref_frame = input_reformatter(out_frame)
                        for new_frame in get_new_frames(frame_callback(ref_frame)):
                            reformatted_frame = reformatter(new_frame)
                            # print(video_input_stream.format, new_frame.format, reformatted_frame.format)
                            enc_packet = video_output_stream.encode(reformatted_frame)
                            if enc_packet:
                                output_container.mux(enc_packet)
                            pbar.update(1)
            elif packet.stream.type == "audio":
                if packet.dts is not None:
                    if audio_copy:
                        packet.stream = audio_output_stream
                        output_container.mux(packet)
                    else:
                        for frame in safe_decode(packet):
                            frame.pts = None
                            enc_packet = cast(av.AudioStream, audio_output_stream).encode(frame)
                            if enc_packet:
                                output_container.mux(enc_packet)

            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break

            if i % 100 == 0:
                gc.collect()

        for frame in fps_filter.flush():
            frame = input_reformatter(frame)
            for new_frame in get_new_frames(frame_callback(frame)):
                ref_frame = reformatter(new_frame)
                enc_packet = video_output_stream.encode(ref_frame)
                if enc_packet:
                    output_container.mux(enc_packet)
                pbar.update(1)

        for new_frame in get_new_frames(frame_callback(None)):
            ref_frame = reformatter(new_frame)
            enc_packet = video_output_stream.encode(ref_frame)
            if enc_packet:
                output_container.mux(enc_packet)
            pbar.update(1)

        packet = video_output_stream.encode(None)
        if packet:
            output_container.mux(packet)

    except KeyboardInterrupt:
        pbar.close()
        output_container.close()
        input_container.close()
        raise
    except:  # noqa
        pbar.close()
        output_container.close()
        input_container.close()
        output_path_error = make_error_file_path(output_path)
        if path.exists(output_path_tmp):
            try_replace(output_path_tmp, output_path_error)
        raise

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

    video_output_stream = output_container.add_stream(config.video_codec, convert_fps_fraction(config.fps))
    configure_colorspace(video_output_stream, None, config)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.width = output_size[0]
    video_output_stream.height = output_size[1]
    video_output_stream.options = config.options
    reformatter = config.state["reformatter"]

    if audio_file is not None:
        input_container = av.open(audio_file, **AV_READ_OPTIONS)
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
                if test_audio_copy(audio_file, output_path):
                    audio_output_stream = output_container.add_stream_from_template(template=audio_input_stream)
                    audio_copy = True
                else:
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
                        for frame in safe_decode(packet):
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


def process_video_keyframes(input_path, frame_callback,
                            min_interval_sec=4., vf="",
                            title=None, stop_event=None, suspend_event=None, tqdm_fn=None):
    sw_format = SoftwareVideoFormat(input_path)
    input_container = av.open(input_path, **AV_READ_OPTIONS)
    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")
    if input_container.duration:
        container_duration = float(input_container.duration / av.time_base)
    else:
        container_duration = None

    video_input_stream = input_container.streams.video[0]
    # video_input_stream.thread_type = "AUTO"  # slow
    video_input_stream.codec_context.skip_frame = "NONKEY"

    video_filter = VideoPreprocessor(video_input_stream, sw_format, vf=vf)

    max_progress = guess_duration(video_input_stream, container_duration=container_duration, input_path=input_path)
    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=max_progress, ncols=ncols)
    prev_sec = 0
    for packet in input_container.demux([video_input_stream]):
        for frame in safe_decode(packet):
            current_sec = math.ceil(frame.pts * video_input_stream.time_base)
            if current_sec - prev_sec >= min_interval_sec:
                for frame in video_filter.update(frame):
                    frame_callback(frame)
                pbar.update(current_sec - prev_sec)
                prev_sec = current_sec
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    for frame in video_filter.flush():
        frame_callback(frame)
        pbar.update(1)

    pbar.close()
    input_container.close()


def hook_frame(
        input_path,
        frame_callback,
        config_callback=default_config_callback,
        title=None,
        vf="",
        stop_event=None, suspend_event=None, tqdm_fn=None,
        start_time=None, end_time=None,
        hwaccel=None,
        disable_software_fallback=False,
        device="cpu",
):
    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")
    if isinstance(device, str):
        device = torch.device(device)

    sw_format = SoftwareVideoFormat(input_path)
    input_hwaccel = create_hwaccel(device=hwaccel, device_id=device.index,
                                   disable_software_fallback=disable_software_fallback)
    input_container = av.open(input_path, **AV_READ_OPTIONS, hwaccel=input_hwaccel)  # type: ignore
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
    config.fps = convert_fps_fraction(config.fps)
    configure_colorspace(None, sw_format, config)
    rgb24_options = config.state["rgb24_options"]
    if should_use_tensor_frame(sw_format.format.name, hwaccel, device):
        input_reformatter = lambda frame: frame
        input_transform = InputTransform(
            src_pix_fmt=sw_format.format.name,
            src_colorspace=rgb24_options["src_colorspace"],
            src_color_primaries=rgb24_options["src_color_primaries"],
            src_color_trc=rgb24_options["src_color_trc"],
            src_color_range=rgb24_options["src_color_range"],
            dst_colorspace=rgb24_options["dst_colorspace"],
            dst_color_primaries=rgb24_options["dst_color_primaries"],
            dst_color_trc=rgb24_options["dst_color_trc"],
            dst_color_range=rgb24_options["dst_color_range"],
            use_16bit=sw_format.use_16bit,
            device=device,
        )
        fps_filter = VideoPreprocessor(video_input_stream, sw_format, fps=config.fps, vf=vf,
                                       input_transform=input_transform)
    else:
        input_reformatter = lambda frame: frame.reformat(
            src_colorspace=rgb24_options["src_colorspace"],
            src_color_range=rgb24_options["src_color_range"],
            dst_colorspace=rgb24_options["dst_colorspace"],
            dst_color_primaries=rgb24_options["dst_color_primaries"],
            dst_color_trc=rgb24_options["dst_color_trc"],
            dst_color_range=rgb24_options["dst_color_range"],
        )
        fps_filter = VideoPreprocessor(video_input_stream, sw_format, fps=config.fps, vf=vf)

    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    total = guess_frames(video_input_stream,
                         fps=config.fps,
                         start_time=start_time, end_time=end_time,
                         container_duration=container_duration,
                         input_path=input_path)
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)

    for i, packet in enumerate(input_container.demux([video_input_stream])):
        if packet.pts is not None:
            if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                break
        for frame in safe_decode(packet, strict=disable_software_fallback):
            for out_frame in fps_filter.update(frame):
                ref_frame = input_reformatter(out_frame)
                frame_callback(ref_frame)
                pbar.update(1)
        if i % 100 == 0:
            gc.collect()
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    for frame in fps_filter.flush():
        ref_frame = input_reformatter(frame)
        frame_callback(ref_frame)
        pbar.update(1)

    frame_callback(None)
    input_container.close()
    pbar.close()


def sample_frames(
        input_path,
        frame_callback,
        num_samples,
        offset=0.05,
        keyframe_only=False,
        vf="",
        title=None,
        stop_event=None,
        suspend_event=None,
        tqdm_fn=None,
        hwaccel=None,
        device="cpu",
        disable_software_fallback=False,
):
    assert offset < 0.5, "offset must be less than 0.5"
    if isinstance(device, str):
        device = torch.device(device)

    sw_format = SoftwareVideoFormat(input_path)
    input_hwaccel = create_hwaccel(device=hwaccel, device_id=device.index,
                                   disable_software_fallback=disable_software_fallback)
    input_container = av.open(input_path, **AV_READ_OPTIONS, hwaccel=input_hwaccel)  # types: ignore

    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    video_input_stream = input_container.streams.video[0]
    if input_container.duration:
        container_duration = float(input_container.duration / av.time_base)
    else:
        container_duration = None

    num_frames, duration = guess_frames(
        video_input_stream,
        container_duration=container_duration,
        input_path=input_path,
        return_duration=True
    )
    if duration <= 0 or num_frames <= 0:
        print(f"sample_frames: No duration available: {input_path}", file=sys.stderr)
        return -1

    max_progress = num_samples
    input_transform = InputTransform(
        src_pix_fmt=sw_format.format.name,
        src_colorspace=sw_format.colorspace,
        src_color_primaries=sw_format.color_primaries,
        src_color_trc=sw_format.color_trc,
        src_color_range=sw_format.color_range,
        dst_colorspace=sw_format.colorspace,
        dst_color_primaries=sw_format.color_primaries,
        dst_color_trc=sw_format.color_trc,
        dst_color_range=ColorRange.JPEG,
        use_16bit=sw_format.use_16bit,
        device=device,
    )
    video_filter = VideoPreprocessor(video_input_stream, sw_format, fps=None, vf=vf,
                                     input_transform=input_transform)

    desc = (title if title else input_path)
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=max_progress, ncols=ncols)
    prev_sec = 0
    sample_count = 0

    if num_samples * 4 > num_frames or duration < num_samples:
        # Full decoding
        step_sec = duration / num_samples
        if keyframe_only:
            video_input_stream.codec_context.skip_frame = "NONKEY"

        for packet in input_container.demux([video_input_stream]):
            for frame in safe_decode(packet):
                if frame.pts is None:
                    continue
                current_sec = float(frame.pts * packet.time_base)
                if current_sec - prev_sec >= step_sec:
                    for frame in video_filter.update(frame):
                        frame_callback(frame)
                        pbar.update(1)
                        sample_count += 1
                    prev_sec = current_sec
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
    else:
        # Sampling

        prev_seek_pos = 0
        step_sec = duration * (1.0 - offset * 2) / num_samples
        seek_offset = int(duration * offset * av.time_base)

        if step_sec > 4 or keyframe_only:
            video_input_stream.codec_context.skip_frame = "NONKEY"
            keyframe_only = True
        else:
            keyframe_only = False

        def sample_one():
            nonlocal prev_sec
            for packet in input_container.demux([video_input_stream]):
                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
                for frame in safe_decode(packet):
                    if frame.pts is None:
                        continue
                    current_sec = float(frame.pts * packet.time_base)
                    if current_sec <= prev_sec:
                        # Seek loop detected
                        return 0
                    if not keyframe_only and current_sec - prev_sec < step_sec:
                        continue

                    for frame in video_filter.update(frame):
                        frame_callback(frame)
                    pbar.update(1)
                    prev_sec = current_sec
                    return 1
            return 0

        for i in range(num_samples):
            pos = int(i * step_sec * av.time_base) + seek_offset
            if pos == 0 or pos == prev_seek_pos:
                pos += av.time_base
            input_container.seek(pos, backward=True, any_frame=False)
            prev_seek_pos = pos
            sample_count += sample_one()
            if stop_event is not None and stop_event.is_set():
                break

    for frame in video_filter.flush():
        frame_callback(frame)
        pbar.update(1)
        sample_count += 1

    pbar.close()
    input_container.close()

    return sample_count


def export_audio(input_path, output_path, start_time=None, end_time=None,
                 title=None, stop_event=None, suspend_event=None, tqdm_fn=None):

    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")

    input_container = av.open(input_path, **AV_READ_OPTIONS)
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
        if test_audio_copy(input_path, output_path):
            audio_output_stream = output_container.add_stream_from_template(template=audio_input_stream)
            audio_copy = True
        else:
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
                for frame in safe_decode(packet):
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
                        choices=["yuv420p", "yuv444p",
                                 "yuv420p10le",
                                 "rgb24", "gbrp16le"],
                        help="colorspace")
    parser.add_argument("--colorspace", type=str, default="auto",
                        choices=["auto", "unspecified", "bt709", "bt709-pc", "bt709-tv", "bt601", "bt601-pc", "bt601-tv",
                                 "bt2020-tv", "bt2020-pq-tv"],
                        help="colorspace")
    parser.add_argument("--video-codec", type=str, default=LIBH264,
                        choices=["libx264", "libopenh264", "libx265",
                                 "h264_nvenc", "hevc_nvenc",
                                 "h264_qsv", "hevc_qsv",
                                 "utvideo", "ffv1"],
                        help="video codec")
    parser.add_argument("--max-workers", type=int, default=0, help="max worker threads")
    parser.add_argument("--gpu", type=int, default=0, help="0: gpu, -1: cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--vf", type=str, default="", help="video filter")
    parser.add_argument("--half-sbs", action="store_true", help="output 1/2 resolution")
    parser.add_argument("--hwaccel", type=str, default=None, choices=get_supported_hwdevices(),
                        help="hwaccel for decode")

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
        frames = torch.cat([frames, frames], dim=3)
        if args.half_sbs:
            # width x 1
            frames = torch.nn.functional.interpolate(frames, size=(frames.shape[-1] // 2, frames.shape[-2]),
                                                     mode="bilinear", align_corners=False)
        for frame in frames:
            yield frame

    use_16bit = pix_fmt_requires_16bit(args.pix_fmt)
    callback = FrameCallbackPool(process_image, batch_size=args.batch_size,
                                 device=device, max_workers=args.max_workers,
                                 max_batch_queue=args.max_workers,
                                 use_16bit=use_16bit)

    process_video(
        args.input,
        args.output, config_callback=make_config,
        frame_callback=callback,
        vf=args.vf,
        device=device,
        hwaccel=args.hwaccel,
    )


def _test_sample_frames():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="output dir")
    parser.add_argument("--num-samples", "-n", type=int, required=True, help="number of samples")
    parser.add_argument("--vf", type=str, default="", help="video filter")
    parser.add_argument("--offset", type=float, default=0.05, help="skip offset")

    args = parser.parse_args()
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
    processed_count = 0

    def callback(frame):
        nonlocal processed_count
        processed_count += 1
        if args.output is not None:
            frame.to_image().save(path.join(args.output, f"{processed_count}.png"))
        return

    sample_count = sample_frames(args.input, frame_callback=callback, num_samples=args.num_samples, vf=args.vf, offset=args.offset)
    print("sample_count", sample_count, "processed_count", processed_count)


if __name__ == "__main__":
    # _test_process_video()
    # _test_export_audio()
    # _test_sample_frames()
    _test_reencode()
