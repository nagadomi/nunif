import argparse
import math
import os
from typing import cast

import av
import av.video.codeccontext
import av.video.stream
import numpy as np
from av.video.reformatter import ColorRange, Colorspace
from PIL import Image

from tests.helpers.smoke_data import (
    add_sine_audio_stream,
    apply_color_metadata,
    generate_rgba_video,
    mux_sine_audio,
)


def generate_concatenated_video(
    codec_name: str,
    output_path: str,
    pix_fmt: str,
    colorspace: str,
    primaries: str,
    trc: str,
    color_range: str,
    scene_filters: list[str],
    scene_duration: float = 1.0,
    fps: int = 30,
) -> None:
    options: dict[str, str] = {}
    if codec_name == "h264":
        vcodec = "libx264"
        options = {"crf": "16", "preset": "superfast"}
    elif codec_name == "hevc":
        vcodec = "libx265"
        options = {"crf": "16", "preset": "superfast", "x265-params": "log-level=error"}
    else:
        raise ValueError(f"Unknown codec: {codec_name}")

    total_duration = scene_duration * len(scene_filters)
    frames_per_scene = int(scene_duration * fps)

    print(f"Generating [{codec_name}]: {os.path.basename(output_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with av.open(output_path, mode="w", format="matroska") as out_container:
        v_stream: av.video.stream.VideoStream | None = None
        a_stream = add_sine_audio_stream(out_container)
        global_frame_idx = 0

        for scene_idx, flt in enumerate(scene_filters):
            with av.open(flt, format="lavfi") as v_in:
                v_frames = v_in.decode(video=0)
                if scene_idx == 0:
                    first_frame = cast(av.VideoFrame, next(v_frames))
                    v_stream = cast(av.video.stream.VideoStream, out_container.add_stream(vcodec, rate=fps))
                    v_stream.width = first_frame.width
                    v_stream.height = first_frame.height
                    v_stream.pix_fmt = pix_fmt
                    v_stream.options = dict(options)

                    v_ctx = cast(av.video.codeccontext.VideoCodecContext, v_stream.codec_context)
                    apply_color_metadata(v_ctx, colorspace, primaries, trc, color_range)

                    # write first frame
                    f_fmt = first_frame.reformat(format=pix_fmt)
                    f_fmt.pts = global_frame_idx
                    for packet in v_stream.encode(f_fmt):
                        out_container.mux(packet)
                    global_frame_idx += 1
                    start_frame = 1
                else:
                    start_frame = 0

                assert v_stream is not None
                for i, frame in enumerate(v_frames, start=start_frame):
                    if i >= frames_per_scene:
                        break
                    v_f = cast(av.VideoFrame, frame)
                    f_fmt = v_f.reformat(format=pix_fmt)
                    f_fmt.pts = global_frame_idx
                    for packet in v_stream.encode(f_fmt):
                        out_container.mux(packet)
                    global_frame_idx += 1

        assert v_stream is not None
        for packet in v_stream.encode(None):
            out_container.mux(packet)

        mux_sine_audio(out_container, a_stream, total_duration)


def generate_image_scenes_video(
    image_paths: list[str],
    output_path: str,
    width: int = 640,
    height: int = 360,
    scene_duration: float = 1.0,
    fps: int = 30,
) -> None:
    total_duration = scene_duration * len(image_paths)
    frames_per_scene = int(scene_duration * fps)

    print(f"Generating [h264]: {os.path.basename(output_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with av.open(output_path, mode="w", format="matroska") as out_container:
        v_stream = cast(av.video.stream.VideoStream, out_container.add_stream("libx264", rate=fps))
        v_stream.width = width
        v_stream.height = height
        v_stream.pix_fmt = "yuv420p"
        v_stream.options = {"crf": "16", "preset": "superfast"}

        v_ctx = cast(av.video.codeccontext.VideoCodecContext, v_stream.codec_context)
        apply_color_metadata(v_ctx, colorspace="bt709", primaries="bt709", trc="bt709", color_range="tv")

        a_stream = add_sine_audio_stream(out_container)
        canvas_w = width + 80
        canvas_h = height + 80
        global_frame_idx = 0

        for image_path in image_paths:
            with Image.open(image_path) as img:
                resized = img.convert("RGB").resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
            for i in range(frames_per_scene):
                t = i / max(1, frames_per_scene - 1)
                x0 = int(round((canvas_w - width) * t))
                y0 = int(round((canvas_h - height) * 0.5 + 20 * math.sin(2 * math.pi * t)))
                crop = resized.crop((x0, y0, x0 + width, y0 + height))
                frame = av.VideoFrame.from_ndarray(np.array(crop), format="rgb24")
                f_fmt = frame.reformat(
                    format="yuv420p",
                    dst_colorspace=Colorspace.ITU709,
                    dst_color_range=ColorRange.MPEG,
                )
                f_fmt.pts = global_frame_idx
                for packet in v_stream.encode(f_fmt):
                    out_container.mux(packet)
                global_frame_idx += 1

        for packet in v_stream.encode(None):
            out_container.mux(packet)

        mux_sine_audio(out_container, a_stream, total_duration)


def copy_video_only(input_path: str, output_path: str) -> None:
    print(f"Generating (no audio): {os.path.basename(output_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with (
        av.open(input_path) as in_container,
        av.open(output_path, mode="w", format="matroska") as out_container,
    ):
        in_video = in_container.streams.video[0]
        out_video = cast(av.video.stream.VideoStream, out_container.add_stream_from_template(in_video))
        for packet in in_container.demux(in_video):
            if packet.dts is None and packet.pts is None:
                continue
            packet.stream = out_video
            out_container.mux(packet)


def ensure_race_condition_data(
    base_dir: str = "tmp/race_condition",
    scene_duration: float = 1.0,
    force: bool = False,
) -> None:
    images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    sd_path = os.path.join(base_dir, "sd.mkv")
    if force or not os.path.exists(sd_path):
        sd_images = [
            os.path.join(images_dir, "dog_448.png"),
            os.path.join(images_dir, "lighthouse_448.png"),
            os.path.join(images_dir, "bottle_448.png"),
        ]
        generate_image_scenes_video(
            image_paths=sd_images,
            output_path=sd_path,
            width=640,
            height=360,
            scene_duration=scene_duration,
        )

    hdr_path = os.path.join(base_dir, "hdr.mkv")
    if force or not os.path.exists(hdr_path):
        scenes_hdr = [
            "testsrc2=size=1280x720:rate=30,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv",
            "smptebars=size=1280x720:rate=30,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv",
            "mandelbrot=size=1280x720:rate=30,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv",
        ]
        generate_concatenated_video(
            codec_name="hevc",
            output_path=hdr_path,
            pix_fmt="yuv420p10le",
            colorspace="bt2020nc",
            primaries="bt2020",
            trc="smpte2084",
            color_range="tv",
            scene_filters=scenes_hdr,
            scene_duration=scene_duration,
        )

    sd_noaudio_path = os.path.join(base_dir, "sd_noaudio.mkv")
    if force or not os.path.exists(sd_noaudio_path):
        copy_video_only(sd_path, sd_noaudio_path)

    hdr_noaudio_path = os.path.join(base_dir, "hdr_noaudio.mkv")
    if force or not os.path.exists(hdr_noaudio_path):
        copy_video_only(hdr_path, hdr_noaudio_path)

    rgba_path = os.path.join(base_dir, "rgba.mkv")
    if force or not os.path.exists(rgba_path):
        rgba_src = os.path.join(images_dir, "dog_448_rgba.png")
        if os.path.exists(rgba_src):
            generate_rgba_video(rgba_src, rgba_path, codec="ffv1", duration=scene_duration * 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate race condition test video datasets")
    parser.add_argument("--base-dir", type=str, default="tmp/race_condition", help="Output directory")
    parser.add_argument("--scene-duration", type=float, default=1.0, help="Per-scene duration in seconds")
    parser.add_argument("--force", action="store_true", help="Force regenerate files")
    args = parser.parse_args()

    ensure_race_condition_data(
        base_dir=args.base_dir,
        scene_duration=args.scene_duration,
        force=args.force,
    )


if __name__ == "__main__":
    main()
