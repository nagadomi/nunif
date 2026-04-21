import gc
import io
import math
import os
import sys
import time
from os import path
from types import GeneratorType
from typing import List

import av
import torch
from av.codec.hwaccel import HWAccel
from tqdm import tqdm

from .color_transform import (
    OutputTransform,
    configure_video_codec,
    setup_color_transform,
)
from .hwaccel import HW_DEVICES, create_hwaccel
from .metadata import (
    AudioMetadata,
    VideoMetadata,
    convert_fps_fraction,
    parse_time,
)
from .output_config import VideoOutputConfig
from .utils import (
    LIBH264,
    get_default_video_codec,
    is_nvidia_gpu,
    pix_fmt_requires_16bit,
)
from .video_preprocessor import VideoPreprocessor


def _print_len(stream):
    sw_format = VideoMetadata.from_file(stream.container.name)
    print("frames", sw_format.stream_frames)
    print("guessed_frames", sw_format.guess_frames())
    print("duration", sw_format.get_duration())
    print("base_rate", float(stream.base_rate))
    print("average_rate", float(stream.average_rate))
    print("guessed_rate", float(stream.guessed_rate))


def default_config_callback(metadata):
    fps = metadata.get_fps()
    if float(fps) > 30:
        fps = 30
    return VideoOutputConfig(fps=fps, options={"preset": "ultrafast", "crf": "20"})


def get_new_frames(frames):
    if frames is None:
        return []
    elif isinstance(frames, (list, tuple)):
        return frames
    elif isinstance(frames, GeneratorType):
        return frames
    else:
        return [frames]


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
        except:  # noqa
            time.sleep(2)
            try_count -= 1
            if try_count <= 0:
                raise


def test_audio_copy(input_path, output_path):
    buff = io.BytesIO()
    buff.name = path.basename(output_path)
    try:
        with (
            av.open(input_path, mode="r", metadata_errors="ignore") as input_container,
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


def fix_frame_color_av17(frame: av.VideoFrame, sw_format: VideoMetadata) -> av.VideoFrame:
    # In av17, VideoFrame color properties are lost when HWAccel(is_hw_owned=False),
    # so restore them from the source properties.
    if frame.colorspace == 2 and frame.color_primaries == 2 and frame.color_trc == 2 and frame.color_range == 0:
        frame.colorspace = sw_format.colorspace
        frame.color_primaries = sw_format.color_primaries
        frame.color_trc = sw_format.color_trc
        frame.color_range = sw_format.color_range

    return frame


def apply_color_settings(output_stream: av.video.stream.VideoStream, output_reformatter: OutputTransform) -> None:
    ctx = output_stream.codec_context
    ctx.pix_fmt = output_reformatter.dst_pix_fmt
    ctx.colorspace = output_reformatter.dst_colorspace
    ctx.color_primaries = output_reformatter.dst_color_primaries
    ctx.color_trc = output_reformatter.dst_color_trc
    ctx.color_range = output_reformatter.dst_color_range


def process_video(
    input_path,
    output_path,
    frame_callback,
    config_callback=default_config_callback,
    title=None,
    vf="",
    stop_event=None,
    suspend_event=None,
    tqdm_fn=None,
    start_time=None,
    end_time=None,
    device: str | torch.device = "cpu",
    inference_mode: bool = True,
    hwaccel: str | None = None,
    disable_software_fallback: bool = False,
):
    with torch.inference_mode(inference_mode):
        _process_video(
            input_path,
            output_path,
            frame_callback,
            config_callback=config_callback,
            title=title,
            vf=vf,
            stop_event=stop_event,
            suspend_event=suspend_event,
            tqdm_fn=tqdm_fn,
            start_time=start_time,
            end_time=end_time,
            device=device,
            hwaccel=hwaccel,
            disable_software_fallback=disable_software_fallback,
        )


def set_output_size_and_flash(container, stream, frame, unmux_packets):
    stream.width = frame.width
    stream.height = frame.height
    for enc_packet in unmux_packets:
        container.mux(enc_packet)
    unmux_packets.clear()


def _process_video(
    input_path,
    output_path,
    frame_callback,
    config_callback,
    title,
    vf,
    stop_event,
    suspend_event,
    tqdm_fn,
    start_time,
    end_time,
    device: str | torch.device,
    hwaccel: str | None,
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

    sw_format = VideoMetadata.from_file(input_path)
    input_hwaccel = create_hwaccel(
        device=hwaccel, device_id=device.index, disable_software_fallback=disable_software_fallback
    )
    output_path_tmp = make_temporary_file_path(output_path)
    input_container = av.open(input_path, mode="r", metadata_errors="ignore", hwaccel=input_hwaccel)

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

    config = config_callback(sw_format)
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
        if is_nvidia_gpu(device):
            device_id = device.index if device.index is not None else 0
        else:
            device_id = None
        output_hwaccel = HWAccel(device_type="cuda", device=device_id, options={"primary_ctx": "1"})
    output_container = av.open(output_path_tmp, mode="w", options=config.container_options, hwaccel=output_hwaccel)

    output_fps = config.output_fps or config.fps
    input_reformat_options, output_reformatter = setup_color_transform(sw_format, config)
    config.pix_fmt = output_reformatter.dst_pix_fmt
    config.output_colorspace = int(output_reformatter.dst_colorspace)
    config.output_color_primaries = int(output_reformatter.dst_color_primaries)
    config.output_color_trc = int(output_reformatter.dst_color_trc)
    config.source_color_range = int(input_reformat_options["src_color_range"])

    if config.state_updated is not None:
        config.state_updated(config)

    video_output_stream = output_container.add_stream(config.video_codec, output_fps)
    apply_color_settings(video_output_stream, output_reformatter)

    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.options = config.options
    video_preprocessor = VideoPreprocessor(
        stream_pix_fmt=video_input_stream.pix_fmt,
        sw_format=sw_format,
        output_colorspace_mode=config.colorspace,
        fps=config.fps,
        vf=vf,
        hwaccel=hwaccel,
        device=device,
        input_reformat_options=input_reformat_options,
    )

    uninitialized: bool = True
    unmux_packets: List[av.Packet | List[av.Packet]] = []

    if config.output_width is not None and config.output_height is not None:
        video_output_stream.width = config.output_width
        video_output_stream.height = config.output_height
        uninitialized = False

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

    desc = title if title else input_path
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    # TODO: `total` may be less when start_time is specified
    total = sw_format.guess_frames(
        fps=output_fps,
        start_time=start_time,
        end_time=end_time,
    )
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]

    try:
        for i, packet in enumerate(input_container.demux(streams)):
            if packet.pts is not None:
                if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                    break
            if packet.stream.type == "video":
                for frame in safe_decode(packet, strict=disable_software_fallback):
                    frame = fix_frame_color_av17(frame, sw_format)
                    for out_frame in video_preprocessor.update(frame):
                        for new_frame in get_new_frames(frame_callback(out_frame)):
                            reformatted_frame = output_reformatter(new_frame)
                            if uninitialized:
                                set_output_size_and_flash(
                                    output_container, video_output_stream, reformatted_frame, unmux_packets
                                )
                                uninitialized = False
                            # print(video_input_stream.format, new_frame.format, reformatted_frame.format)
                            enc_packets = video_output_stream.encode(reformatted_frame)
                            if enc_packets:
                                output_container.mux(enc_packets)
                            pbar.update(1)
            elif packet.stream.type == "audio":
                assert isinstance(audio_output_stream, av.AudioStream)
                if packet.dts is not None:
                    if audio_copy:
                        packet.stream = audio_output_stream
                        if uninitialized:
                            unmux_packets.append(packet)
                        else:
                            output_container.mux(packet)
                    else:
                        for frame in safe_decode(packet):
                            frame.pts = None
                            enc_packets = audio_output_stream.encode(frame)
                            if enc_packets:
                                if uninitialized:
                                    unmux_packets.append(enc_packets)
                                else:
                                    output_container.mux(enc_packets)

            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break

            if i % 100 == 0:
                gc.collect()

        for out_frame in video_preprocessor.flush():
            for new_frame in get_new_frames(frame_callback(out_frame)):
                ref_frame = output_reformatter(new_frame)
                if uninitialized:
                    set_output_size_and_flash(output_container, video_output_stream, ref_frame, unmux_packets)
                    uninitialized = False

                enc_packets = video_output_stream.encode(ref_frame)
                if enc_packets:
                    output_container.mux(enc_packets)
                pbar.update(1)

        for new_frame in get_new_frames(frame_callback(None)):
            ref_frame = output_reformatter(new_frame)
            if uninitialized:
                set_output_size_and_flash(output_container, video_output_stream, ref_frame, unmux_packets)
                uninitialized = False
            enc_packets = video_output_stream.encode(ref_frame)
            if enc_packets:
                output_container.mux(enc_packets)
            pbar.update(1)

        enc_packets = video_output_stream.encode(None)
        if enc_packets:
            output_container.mux(enc_packets)

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


def generate_video(
    output_path,
    frame_generator,
    config,
    audio_file=None,
    title=None,
    total_frames=None,
    stop_event=None,
    suspend_event=None,
    tqdm_fn=None,
):

    output_path_tmp = path.join(path.dirname(output_path), "_tmp_" + path.basename(output_path))
    output_container = av.open(output_path_tmp, "w", options=config.container_options)
    output_size = config.output_width, config.output_height

    if not config.container_format:
        config.container_format = path.splitext(output_path)[-1].lower()[1:]
    if not config.video_codec:
        config.video_codec = get_default_video_codec(config.container_format)
    configure_video_codec(config)

    input_reformat_options, output_reformatter = setup_color_transform(None, config)
    config.pix_fmt = output_reformatter.dst_pix_fmt
    config.output_colorspace = int(output_reformatter.dst_colorspace)
    config.output_color_primaries = int(output_reformatter.dst_color_primaries)
    config.output_color_trc = int(output_reformatter.dst_color_trc)
    config.source_color_range = int(input_reformat_options["src_color_range"])

    if config.state_updated is not None:
        config.state_updated(config)

    video_output_stream = output_container.add_stream(config.video_codec, convert_fps_fraction(config.fps))
    apply_color_settings(video_output_stream, output_reformatter)

    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.width = output_size[0]
    video_output_stream.height = output_size[1]
    video_output_stream.options = config.options

    if audio_file is not None:
        input_container = av.open(audio_file, mode="r", metadata_errors="ignore")
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
            desc = title + " Audio" if title else "Audio"
            ncols = len(desc) + 60
            sw_audio = AudioMetadata.from_file(audio_file)
            total = sw_audio.get_duration()
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

    desc = title + " Frames" if title else "Frames"
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=total_frames, ncols=ncols)
    for frame in frame_generator():
        if frame is None:
            break
        for new_frame in get_new_frames(frame):
            new_frame = output_reformatter(new_frame)
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


def consume_generator(callback_result):
    if isinstance(callback_result, GeneratorType):
        for _ in callback_result:
            pass


def process_video_keyframes(
    input_path,
    frame_callback,
    min_interval_sec=4.0,
    vf="",
    title=None,
    stop_event=None,
    suspend_event=None,
    tqdm_fn=None,
):
    sw_format = VideoMetadata.from_file(input_path)
    input_container = av.open(input_path, mode="r", metadata_errors="ignore")
    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    video_input_stream = input_container.streams.video[0]
    # video_input_stream.thread_type = "AUTO"  # slow
    video_input_stream.codec_context.skip_frame = "NONKEY"

    video_filter = VideoPreprocessor(
        video_input_stream.pix_fmt, sw_format, output_colorspace_mode="auto", fps=None, vf=vf
    )

    max_progress = sw_format.guess_duration()
    desc = title if title else input_path
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=max_progress, ncols=ncols)
    prev_sec = 0
    for packet in input_container.demux([video_input_stream]):
        for frame in safe_decode(packet):
            frame = fix_frame_color_av17(frame, sw_format)
            current_sec = math.ceil(frame.pts * video_input_stream.time_base)
            if current_sec - prev_sec >= min_interval_sec:
                for frame in video_filter.update(frame):
                    consume_generator(frame_callback(frame))
                pbar.update(current_sec - prev_sec)
                prev_sec = current_sec
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    for frame in video_filter.flush():
        consume_generator(frame_callback(frame))
        pbar.update(1)

    pbar.close()
    input_container.close()


def hook_frame(
    input_path,
    frame_callback,
    config_callback=default_config_callback,
    title=None,
    vf="",
    stop_event=None,
    suspend_event=None,
    tqdm_fn=None,
    start_time=None,
    end_time=None,
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

    sw_format = VideoMetadata.from_file(input_path)
    input_hwaccel = create_hwaccel(
        device=hwaccel, device_id=device.index, disable_software_fallback=disable_software_fallback
    )
    input_container = av.open(input_path, mode="r", metadata_errors="ignore", hwaccel=input_hwaccel)

    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    if start_time is not None:
        input_container.seek(start_time * av.time_base, backward=True, any_frame=False)

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"

    config = config_callback(sw_format)
    config.fps = convert_fps_fraction(config.fps)
    input_reformat_options, output_reformatter = setup_color_transform(sw_format, config)
    config.pix_fmt = output_reformatter.dst_pix_fmt
    config.output_colorspace = int(output_reformatter.dst_colorspace)
    config.output_color_primaries = int(output_reformatter.dst_color_primaries)
    config.output_color_trc = int(output_reformatter.dst_color_trc)
    config.source_color_range = int(input_reformat_options["src_color_range"])

    if config.state_updated is not None:
        config.state_updated(config)

    video_preprocessor = VideoPreprocessor(
        stream_pix_fmt=video_input_stream.pix_fmt,
        sw_format=sw_format,
        output_colorspace_mode=config.colorspace,
        fps=config.fps,
        vf=vf,
        hwaccel=hwaccel,
        device=device,
        input_reformat_options=input_reformat_options,
    )

    desc = title if title else input_path
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    total = sw_format.guess_frames(
        fps=config.fps,
        start_time=start_time,
        end_time=end_time,
    )
    pbar = tqdm_fn(desc=desc, total=total, ncols=ncols)

    for i, packet in enumerate(input_container.demux([video_input_stream])):
        if packet.pts is not None:
            if end_time is not None and packet.stream.type == "video" and end_time < packet.pts * packet.time_base:
                break
        for frame in safe_decode(packet, strict=disable_software_fallback):
            frame = fix_frame_color_av17(frame, sw_format)
            for out_frame in video_preprocessor.update(frame):
                consume_generator(frame_callback(out_frame))
                pbar.update(1)
        if i % 100 == 0:
            gc.collect()
        if suspend_event is not None:
            suspend_event.wait()
        if stop_event is not None and stop_event.is_set():
            break

    for out_frame in video_preprocessor.flush():
        consume_generator(frame_callback(out_frame))
        pbar.update(1)

    consume_generator(frame_callback(None))
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

    sw_format = VideoMetadata.from_file(input_path)
    input_hwaccel = create_hwaccel(
        device=hwaccel, device_id=device.index, disable_software_fallback=disable_software_fallback
    )
    input_container = av.open(input_path, mode="r", metadata_errors="ignore", hwaccel=input_hwaccel)  # types: ignore

    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    video_input_stream = input_container.streams.video[0]

    num_frames, duration = sw_format.guess_frames(return_duration=True)
    if duration <= 0 or num_frames <= 0:
        print(f"sample_frames: No duration available: {input_path}", file=sys.stderr)
        return -1

    video_preprocessor = VideoPreprocessor(
        stream_pix_fmt=video_input_stream.pix_fmt,
        sw_format=sw_format,
        output_colorspace_mode="auto",
        fps=None,
        vf=vf,
        hwaccel=hwaccel,
        device=device,
        input_reformat_options=sw_format.get_auto_input_reformat_options(),
    )

    max_progress = num_samples
    desc = title if title else input_path
    ncols = len(desc) + 60
    tqdm_fn = tqdm_fn or tqdm
    pbar = tqdm_fn(desc=desc, total=max_progress, ncols=ncols)
    prev_sec = 0
    sample_count = 0
    packet_count = 0
    frame_count = 0

    if num_samples * 4 > num_frames or duration < num_samples:
        # Full decoding
        step_sec = duration / num_samples
        if keyframe_only:
            video_input_stream.codec_context.skip_frame = "NONKEY"

        for packet in input_container.demux([video_input_stream]):
            packet_count += 1
            for frame in safe_decode(packet, strict=disable_software_fallback):
                frame_count += 1
                frame = fix_frame_color_av17(frame, sw_format)
                if frame.pts is None:
                    continue
                current_sec = float(frame.pts * packet.time_base)
                if current_sec - prev_sec >= step_sec:
                    for out_frame in video_preprocessor.update(frame):
                        consume_generator(frame_callback(out_frame))
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
            nonlocal prev_sec, packet_count, frame_count
            for packet in input_container.demux([video_input_stream]):
                packet_count += 1
                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
                for frame in safe_decode(packet, strict=disable_software_fallback):
                    frame_count += 1
                    frame = fix_frame_color_av17(frame, sw_format)
                    if frame.pts is None:
                        continue
                    current_sec = float(frame.pts * packet.time_base)
                    if current_sec <= prev_sec:
                        # Seek loop detected
                        return 0
                    if not keyframe_only and current_sec - prev_sec < step_sec:
                        continue

                    for out_frame in video_preprocessor.update(frame):
                        consume_generator(frame_callback(out_frame))
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

    for out_frame in video_preprocessor.flush():
        frame_count += 1
        consume_generator(frame_callback(out_frame))
        pbar.update(1)
        sample_count += 1

    pbar.close()
    input_container.close()

    if frame_count == 0 and packet_count > 2:
        raise RuntimeError(
            f"Unable to sample frames. Likely an HWAccel error. \nframe_count=0, packet_count={packet_count}"
        )

    return sample_count


def export_audio(
    input_path,
    output_path,
    start_time=None,
    end_time=None,
    title=None,
    stop_event=None,
    suspend_event=None,
    tqdm_fn=None,
):
    if isinstance(start_time, str):
        start_time = parse_time(start_time)
    if isinstance(end_time, str):
        end_time = parse_time(end_time)
        if start_time is not None and not (start_time < end_time):
            raise ValueError("end_time must be greater than start_time")

    input_container = av.open(input_path, mode="r", metadata_errors="ignore")
    if len(input_container.streams.audio) == 0:
        input_container.close()
        return False

    if start_time is not None:
        input_container.seek(start_time * av.time_base, backward=True, any_frame=False)

    audio_input_stream = input_container.streams.audio[0]
    output_container = av.open(output_path, "w")  # expect .m4a

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
    sw_format = VideoMetadata.from_file(input_path)
    total = math.ceil(sw_format.get_duration())
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
    import argparse

    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output video file")
    args = parser.parse_args()

    def make_config(metadata):
        fps = metadata.get_fps()
        if fps > 30:
            fps = 30
        return VideoOutputConfig(fps=fps, options={"preset": "ultrafast", "crf": "20"})

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

    from .frame_callback_pool import FrameCallbackPool

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output video file")
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp16le"],
        help="colorspace",
    )
    parser.add_argument(
        "--colorspace",
        type=str,
        default="auto",
        choices=[
            "auto",
            "unspecified",
            "bt709",
            "bt709-pc",
            "bt709-tv",
            "bt601",
            "bt601-pc",
            "bt601-tv",
            "bt2020-tv",
            "bt2020-pq-tv",
        ],
        help="colorspace",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default=LIBH264,
        choices=[
            "libx264",
            "libopenh264",
            "libx265",
            "h264_nvenc",
            "hevc_nvenc",
            "h264_qsv",
            "hevc_qsv",
            "utvideo",
            "ffv1",
        ],
        help="video codec",
    )
    parser.add_argument("--max-workers", type=int, default=0, help="max worker threads")
    parser.add_argument("--gpu", type=int, default=0, help="0: gpu, -1: cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--vf", type=str, default="", help="video filter")
    parser.add_argument("--half-sbs", action="store_true", help="output 1/2 resolution")
    parser.add_argument("--hwaccel", type=str, default=None, choices=HW_DEVICES, help="hwaccel for decode")

    args = parser.parse_args()
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    preset = "fast" if args.video_codec in {"h264_nvenc", "hevc_nvenc"} else "ultrafast"

    def make_config(metadata):
        fps = metadata.get_fps()
        if fps > 30:
            fps = 30
        return VideoOutputConfig(
            fps=fps,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            video_codec=args.video_codec,
            options={"preset": preset, "crf": "20"},
        )

    def process_image(frames):
        # width x 2
        frames = torch.cat([frames, frames], dim=3)
        if args.half_sbs:
            # width x 1
            frames = torch.nn.functional.interpolate(
                frames, size=(frames.shape[-1] // 2, frames.shape[-2]), mode="bilinear", align_corners=False
            )
        for frame in frames:
            yield frame

    use_16bit = pix_fmt_requires_16bit(args.pix_fmt)
    callback = FrameCallbackPool(
        process_image,
        batch_size=args.batch_size,
        device=device,
        max_workers=args.max_workers,
        max_batch_queue=args.max_workers,
        use_16bit=use_16bit,
    )

    process_video(
        args.input,
        args.output,
        config_callback=make_config,
        frame_callback=callback,
        vf=args.vf,
        device=device,
        hwaccel=args.hwaccel,
    )


def _test_sample_frames():
    import argparse

    import torchvision.transforms.functional as TF

    from .frame_callback_pool import to_tensor

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="output dir")
    parser.add_argument("--num-samples", "-n", type=int, required=True, help="number of samples")
    parser.add_argument("--vf", type=str, default="", help="video filter")
    parser.add_argument("--offset", type=float, default=0.05, help="skip offset")
    parser.add_argument("--hwaccel", type=str, default=None, choices=HW_DEVICES, help="hwaccel for decode")
    parser.add_argument("--gpu", type=int, default=0, help="0: gpu, -1: cpu")

    args = parser.parse_args()
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
    processed_count = 0

    def callback(frame):
        nonlocal processed_count
        processed_count += 1
        if args.output is not None:
            TF.to_pil_image(to_tensor(frame)).save(path.join(args.output, f"{processed_count}.png"))
        return

    sample_count = sample_frames(
        args.input,
        frame_callback=callback,
        num_samples=args.num_samples,
        vf=args.vf,
        offset=args.offset,
        hwaccel=args.hwaccel,
        device=device,
        disable_software_fallback=True,
    )
    print("sample_count", sample_count, "processed_count", processed_count)


if __name__ == "__main__":
    from . import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context()
    # _test_process_video()
    # _test_export_audio()
    # _test_sample_frames()
    _test_reencode()
