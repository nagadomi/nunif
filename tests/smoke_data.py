import argparse
import os
import sys
from collections.abc import Mapping
from typing import NamedTuple, cast

import av
import av.audio.stream
import av.video.codeccontext
import av.video.stream
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc
from PIL import Image

from nunif.utils.video import to_ndarray

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


class VideoTarget(NamedTuple):
    filename: str
    codec: str
    pix_fmt: str
    colorspace: str
    primaries: str
    trc: str
    color_range: str
    input_filter: str
    width: int
    height: int


VIDEO_TARGETS: list[VideoTarget] = [
    VideoTarget(
        filename="sd.mkv",
        codec="h264",
        pix_fmt="yuv420p",
        colorspace="bt709",
        primaries="bt709",
        trc="bt709",
        color_range="tv",
        input_filter="gradients=size=640x360:rate=30:n=8:seed=1",
        width=640,
        height=360,
    ),
]

for _pix_fmt in ("yuv444p", "yuv422p", "gbrp"):
    VIDEO_TARGETS.append(
        VideoTarget(
            filename=f"h264_{_pix_fmt}.mkv",
            codec="h264",
            pix_fmt=_pix_fmt,
            colorspace="bt709",
            primaries="bt709",
            trc="bt709",
            color_range="tv",
            input_filter="gradients=size=640x360:rate=30:n=8:seed=1",
            width=640,
            height=360,
        )
    )
    VIDEO_TARGETS.append(
        VideoTarget(
            filename=f"hevc_{_pix_fmt}.mkv",
            codec="hevc",
            pix_fmt=_pix_fmt,
            colorspace="bt709",
            primaries="bt709",
            trc="bt709",
            color_range="tv",
            input_filter="gradients=size=640x360:rate=30:n=8:seed=1",
            width=640,
            height=360,
        )
    )

VIDEO_TARGETS.append(
    VideoTarget(
        filename="hdr.mkv",
        codec="hevc",
        pix_fmt="yuv420p10le",
        colorspace="bt2020nc",
        primaries="bt2020",
        trc="smpte2084",
        color_range="tv",
        input_filter="gradients=size=1280x720:rate=30:n=8:seed=1",
        width=1280,
        height=720,
    )
)


def generate_video(
    target: VideoTarget,
    output_path: str,
    duration: float = 1.1,
    fps: int = 30,
) -> None:
    options: dict[str, str] = {}
    is_rgb = target.pix_fmt in ("gbrp", "rgb24")
    encode_pix_fmt = target.pix_fmt

    if target.codec == "h264":
        if is_rgb:
            vcodec = "libx264rgb"
            encode_pix_fmt = "rgb24"
            options = {"crf": "16", "preset": "superfast"}
        else:
            vcodec = "libx264"
            options = {"crf": "16", "preset": "superfast", "tune": "fastdecode"}
    elif target.codec == "hevc":
        vcodec = "libx265"
        options = {"crf": "16", "preset": "superfast", "x265-params": "log-level=error"}
    elif target.codec == "ffv1":
        vcodec = "ffv1"
    else:
        raise ValueError(f"Unknown codec: {target.codec}")

    codec_obj = av.Codec(vcodec, "w")
    supported_formats = {f.name for f in codec_obj.video_formats} if codec_obj.video_formats else set()
    if encode_pix_fmt not in supported_formats:
        encode_pix_fmt = "yuv444p"

    sp_parts: list[str] = []
    if is_rgb:
        sp_parts.append("colorspace=gbr")
        sp_parts.append("range=pc")
    else:
        if target.colorspace != "undefined":
            sp_parts.append(f"colorspace={target.colorspace}")
        if target.color_range != "undefined":
            sp_parts.append(f"range={target.color_range}")
    if target.primaries != "undefined":
        sp_parts.append(f"color_primaries={target.primaries}")
    if target.trc != "undefined":
        sp_parts.append(f"color_trc={target.trc}")

    filter_chain = target.input_filter
    if sp_parts:
        filter_chain += f",setparams={':'.join(sp_parts)}"

    print(f"Generating [{target.codec}]: {os.path.basename(output_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with (
        av.open(filter_chain, format="lavfi") as v_in,
        av.open(f"sine=frequency=1000:duration={duration}:sample_rate=44100", format="lavfi") as a_in,
        av.open(output_path, mode="w", format="matroska") as out_container,
    ):
        v_stream = cast(av.video.stream.VideoStream, out_container.add_stream(vcodec, rate=fps))
        first_v_frame = cast(av.VideoFrame, next(v_in.decode(video=0)))
        v_stream.width = first_v_frame.width
        v_stream.height = first_v_frame.height
        v_stream.pix_fmt = encode_pix_fmt
        v_stream.options = dict(options)

        v_ctx = cast(av.video.codeccontext.VideoCodecContext, v_stream.codec_context)
        if is_rgb:
            v_ctx.colorspace = 0
            v_ctx.color_range = ColorRange.JPEG
        else:
            if target.colorspace in COLORSPACE_MAP:
                v_ctx.colorspace = COLORSPACE_MAP[target.colorspace]
            if target.color_range in RANGE_MAP:
                v_ctx.color_range = RANGE_MAP[target.color_range]

        if target.primaries in PRIMARIES_MAP:
            v_ctx.color_primaries = PRIMARIES_MAP[target.primaries]
        if target.trc in TRC_MAP:
            v_ctx.color_trc = TRC_MAP[target.trc]

        a_stream = cast(av.audio.stream.AudioStream, out_container.add_stream("aac", rate=44100))
        a_stream.format = "fltp"
        a_stream.layout = "mono"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=44100)

        max_frames = int(duration * fps)

        # Encode video frames
        f_fmt = first_v_frame.reformat(format=encode_pix_fmt)
        f_fmt.pts = 0
        for packet in v_stream.encode(f_fmt):
            out_container.mux(packet)

        for i, frame in enumerate(v_in.decode(video=0), start=1):
            if i >= max_frames:
                break
            v_f = cast(av.VideoFrame, frame)
            f_fmt = v_f.reformat(format=encode_pix_fmt)
            f_fmt.pts = i
            for packet in v_stream.encode(f_fmt):
                out_container.mux(packet)

        for packet in v_stream.encode(None):
            out_container.mux(packet)

        # Encode audio frames
        for audio_frame in a_in.decode(audio=0):
            a_f = cast(av.AudioFrame, audio_frame)
            for rf in resampler.resample(a_f):
                for packet in a_stream.encode(rf):
                    out_container.mux(packet)

        for packet in a_stream.encode(None):
            out_container.mux(packet)


def generate_image_from_video(video_path: str, image_path: str) -> None:
    print(f"Generating image: {os.path.basename(image_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            nd = to_ndarray(frame)
            img: Image.Image = Image.fromarray(nd)
            img.save(image_path)
            break


def colorspace_to_str(val: int | None) -> str:
    if val is None:
        return "None"
    if val == 0:
        return "RGB"
    try:
        return Colorspace(val).name
    except (ValueError, TypeError):
        return str(val)


def color_primaries_to_str(val: int | None) -> str:
    if val is None:
        return "None"
    try:
        return ColorPrimaries(val).name
    except (ValueError, TypeError):
        return str(val)


def color_trc_to_str(val: int | None) -> str:
    if val is None:
        return "None"
    try:
        return ColorTrc(val).name
    except (ValueError, TypeError):
        return str(val)


def color_range_to_str(val: int | None) -> str:
    if val is None:
        return "None"
    try:
        return ColorRange(val).name
    except (ValueError, TypeError):
        return str(val)


def verify_video(file_path: str, target: VideoTarget) -> bool:
    if not os.path.exists(file_path):
        print(f"[FAIL] {os.path.basename(file_path)}: File not found")
        return False

    is_rgb = target.pix_fmt in ("gbrp", "rgb24")
    expected_cs = 0 if is_rgb else int(COLORSPACE_MAP[target.colorspace])
    expected_pri = int(PRIMARIES_MAP[target.primaries])
    expected_trc = int(TRC_MAP[target.trc])
    expected_range = int(ColorRange.JPEG) if is_rgb else int(RANGE_MAP[target.color_range])

    with av.open(file_path) as container:
        if not container.streams.video:
            print(f"[FAIL] {os.path.basename(file_path)}: No video stream found")
            return False
        if not container.streams.audio:
            print(f"[FAIL] {os.path.basename(file_path)}: No audio stream found")
            return False

        v_stream = container.streams.video[0]
        v_ctx = v_stream.codec_context
        a_stream = container.streams.audio[0]
        a_ctx = a_stream.codec_context

        actual_cs = int(v_ctx.colorspace) if v_ctx.colorspace is not None else None
        actual_pri = int(v_ctx.color_primaries) if v_ctx.color_primaries is not None else None
        actual_trc = int(v_ctx.color_trc) if v_ctx.color_trc is not None else None
        actual_range = int(v_ctx.color_range) if v_ctx.color_range is not None else None

        cs_str = colorspace_to_str(actual_cs)
        pri_str = color_primaries_to_str(actual_pri)
        trc_str = color_trc_to_str(actual_trc)
        range_str = color_range_to_str(actual_range)

        mismatches: list[str] = []
        if v_ctx.name != target.codec:
            mismatches.append(f"codec: {v_ctx.name} != {target.codec}")
        if v_ctx.pix_fmt != target.pix_fmt:
            mismatches.append(f"pix_fmt: {v_ctx.pix_fmt} != {target.pix_fmt}")
        if v_ctx.width != target.width or v_ctx.height != target.height:
            mismatches.append(f"size: {v_ctx.width}x{v_ctx.height} != {target.width}x{target.height}")
        if actual_cs != expected_cs:
            mismatches.append(f"colorspace: {cs_str} != {colorspace_to_str(expected_cs)}")
        if actual_pri != expected_pri:
            mismatches.append(f"color_primaries: {pri_str} != {color_primaries_to_str(expected_pri)}")
        if actual_trc != expected_trc:
            mismatches.append(f"color_trc: {trc_str} != {color_trc_to_str(expected_trc)}")
        if actual_range != expected_range:
            mismatches.append(f"color_range: {range_str} != {color_range_to_str(expected_range)}")
        if a_ctx.name != "aac":
            mismatches.append(f"audio_codec: {a_ctx.name} != aac")
        if a_stream.rate != 44100:
            mismatches.append(f"audio_rate: {a_stream.rate} != 44100")
        if a_stream.channels != 1:
            mismatches.append(f"audio_channels: {a_stream.channels} != 1")

        try:
            first_frame = next(container.decode(v_stream))
            if first_frame.width != target.width or first_frame.height != target.height:
                mismatches.append(
                    f"decoded_frame_size: {first_frame.width}x{first_frame.height} != {target.width}x{target.height}"
                )
        except Exception as e:
            mismatches.append(f"decode_error: {e}")

        name = os.path.basename(file_path)
        if mismatches:
            print(f"[FAIL] {name}: {', '.join(mismatches)}")
            return False
        else:
            print(
                f"[PASS] {name}: {v_ctx.name} {v_ctx.pix_fmt} {v_ctx.width}x{v_ctx.height} "
                f"cs={cs_str} pri={pri_str} trc={trc_str} range={range_str} "
                f"| audio: {a_ctx.name} {a_stream.rate}Hz"
            )
            return True


def verify_image(file_path: str, expected_format: str, expected_size: tuple[int, int], expected_mode: str) -> bool:
    if not os.path.exists(file_path):
        print(f"[FAIL] {os.path.basename(file_path)}: File not found")
        return False

    name = os.path.basename(file_path)
    try:
        with Image.open(file_path) as img:
            mismatches: list[str] = []
            if img.format != expected_format:
                mismatches.append(f"format: {img.format} != {expected_format}")
            if img.size != expected_size:
                mismatches.append(f"size: {img.size} != {expected_size}")
            if img.mode != expected_mode:
                mismatches.append(f"mode: {img.mode} != {expected_mode}")

            if mismatches:
                print(f"[FAIL] {name}: {', '.join(mismatches)}")
                return False
            else:
                print(f"[PASS] {name}: {img.format} {img.size[0]}x{img.size[1]} {img.mode}")
                return True
    except Exception as e:
        print(f"[FAIL] {name}: Error opening image ({e})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate smoke test video and image datasets")
    parser.add_argument("--base-dir", type=str, default="tests/data/smoke", help="Output directory")
    parser.add_argument("--duration", type=float, default=1.1, help="Duration in seconds")
    parser.add_argument("--force", action="store_true", help="Force regenerate files")
    args = parser.parse_args()

    base_dir = args.base_dir
    duration = args.duration
    force = args.force

    # Generate videos
    for target in VIDEO_TARGETS:
        out_path = os.path.join(base_dir, target.filename)
        if force or not os.path.exists(out_path):
            generate_video(target, out_path, duration=duration)

    # Generate image
    sd_path = os.path.join(base_dir, "sd.mkv")
    png_path = os.path.join(base_dir, "sd.png")
    if force or not os.path.exists(png_path):
        generate_image_from_video(sd_path, png_path)

    # Verify generated data
    print("=" * 70)
    print("Verifying smoke test data formats and metadata")
    print("=" * 70)

    failed = False
    for target in VIDEO_TARGETS:
        out_path = os.path.join(base_dir, target.filename)
        if not verify_video(out_path, target):
            failed = True

    if not verify_image(png_path, "PNG", (640, 360), "RGB"):
        failed = True

    print("=" * 70)
    if not failed:
        print("ALL SMOKE DATA VERIFICATIONS PASSED!")
        print("=" * 70)
    else:
        print("SOME SMOKE DATA VERIFICATIONS FAILED!")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
