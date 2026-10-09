import argparse
import os
from collections.abc import Mapping
from typing import cast

import av
import av.audio.stream
import av.video.codeccontext
import av.video.stream
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

COLORSPACE_MAP: Mapping[str, Colorspace] = {
    "bt709": Colorspace.ITU709,
    "bt2020nc": Colorspace.BT2020,
}

PRIMARIES_MAP: Mapping[str, ColorPrimaries] = {
    "bt709": ColorPrimaries.BT709,
    "bt2020": ColorPrimaries.BT2020,
}

TRC_MAP: Mapping[str, ColorTrc] = {
    "bt709": ColorTrc.BT709,
    "smpte2084": ColorTrc.SMPTE2084,
}

RANGE_MAP: Mapping[str, ColorRange] = {
    "tv": ColorRange.MPEG,
    "pc": ColorRange.JPEG,
}


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

    with (
        av.open(f"sine=frequency=1000:duration={total_duration}:sample_rate=44100", format="lavfi") as a_in,
        av.open(output_path, mode="w", format="matroska") as out_container,
    ):
        v_stream: av.video.stream.VideoStream | None = None
        a_stream = cast(av.audio.stream.AudioStream, out_container.add_stream("aac", rate=44100))
        a_stream.format = "fltp"
        a_stream.layout = "mono"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=44100)
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
                    if colorspace in COLORSPACE_MAP:
                        v_ctx.colorspace = COLORSPACE_MAP[colorspace]
                    if color_range in RANGE_MAP:
                        v_ctx.color_range = RANGE_MAP[color_range]
                    if primaries in PRIMARIES_MAP:
                        v_ctx.color_primaries = PRIMARIES_MAP[primaries]
                    if trc in TRC_MAP:
                        v_ctx.color_trc = TRC_MAP[trc]

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

        # Audio stream
        for audio_frame in a_in.decode(audio=0):
            a_f = cast(av.AudioFrame, audio_frame)
            for rf in resampler.resample(a_f):
                for packet in a_stream.encode(rf):
                    out_container.mux(packet)

        for packet in a_stream.encode(None):
            out_container.mux(packet)


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
            if packet.dts is None:
                continue
            packet.stream = out_video
            out_container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate race condition test video datasets")
    parser.add_argument("--base-dir", type=str, default="tests/data/race_condition", help="Output directory")
    parser.add_argument("--scene-duration", type=float, default=1.0, help="Per-scene duration in seconds")
    parser.add_argument("--force", action="store_true", help="Force regenerate files")
    args = parser.parse_args()

    base_dir = args.base_dir
    scene_dur = args.scene_duration
    force = args.force

    sd_path = os.path.join(base_dir, "sd.mkv")
    if force or not os.path.exists(sd_path):
        scenes_sd = [
            "gradients=size=640x360:rate=30:n=8:seed=1,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
            "smptebars=size=640x360:rate=30,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
            "mandelbrot=size=640x360:rate=30,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
        ]
        generate_concatenated_video(
            codec_name="h264",
            output_path=sd_path,
            pix_fmt="yuv420p",
            colorspace="bt709",
            primaries="bt709",
            trc="bt709",
            color_range="tv",
            scene_filters=scenes_sd,
            scene_duration=scene_dur,
        )

    hdr_path = os.path.join(base_dir, "hdr.mkv")
    if force or not os.path.exists(hdr_path):
        scenes_hdr = [
            "gradients=size=1280x720:rate=30:n=8:seed=1,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv",
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
            scene_duration=scene_dur,
        )

    sd_noaudio_path = os.path.join(base_dir, "sd_noaudio.mkv")
    if force or not os.path.exists(sd_noaudio_path):
        copy_video_only(sd_path, sd_noaudio_path)

    hdr_noaudio_path = os.path.join(base_dir, "hdr_noaudio.mkv")
    if force or not os.path.exists(hdr_noaudio_path):
        copy_video_only(hdr_path, hdr_noaudio_path)


if __name__ == "__main__":
    main()
