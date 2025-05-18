# Split video by Shot Boundary Detection (TransNetV2)
# python -m nunif.cli.semgent_video -i input.mp4 -o output_dir
import nunif.utils.video as VU
import av
import nunif.utils.shot_boundary_detection as SBD
import torch
import argparse
import os
from os import path
from tqdm import tqdm
from nunif.device import create_device
from nunif.logger import logger
import math


def open_output_video(output_path, ext, no, video_input_stream, audio_input_stream, args):
    output_container = av.open(path.join(output_path, f"{str(no).zfill(6)}{ext}"), "w")
    video_output_stream = output_container.add_stream(args.codec, rate=video_input_stream.guessed_rate)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = video_input_stream.pix_fmt
    if args.max_height is None or args.max_height >= video_input_stream.height:
        video_output_stream.width = video_input_stream.width
        video_output_stream.height = video_input_stream.height
    else:
        width = math.ceil((args.max_height / video_input_stream.height) * video_input_stream.width)
        if width % 2 != 0:
            width -= 1
        video_output_stream.width = width
        video_output_stream.height = args.max_height

    if args.codec == "libx264":
        video_output_stream.options = {"crf": str(args.crf), "preset": "veryfast"}
    elif args.codec == "libopenh264":
        video_output_stream.options = {"b": "24M"}
    video_output_stream.thread_type = "AUTO"
    if audio_input_stream is not None:
        # TODO: Some formats cause errors
        VU.add_stream_from_template(output_container, template=audio_input_stream)

    return output_container


def close_output_video(output_container, video_filter):
    while True:
        frame = video_filter.update(None)
        if frame is not None:
            enc_packet = output_container.streams.video[0].encode(frame)
            if enc_packet:
                output_container.mux(enc_packet)
        else:
            break

    enc_packet = output_container.streams.video[0].encode(None)
    if enc_packet:
        output_container.mux(enc_packet)

    output_container.close()


def create_video_filter(video_input_stream, args):
    if args.max_height is None or args.max_height >= video_input_stream.height:
        return VU.VideoFilter(video_input_stream, vf="")  # dummy
    else:
        return VU.VideoFilter(video_input_stream, vf=f"scale=-2:{args.max_height}:flags=bilinear")


def segment_video(pts, args):
    if args.codec == "ffv1":
        ext = ".mkv"
    else:
        ext = ".mp4"
    input_container = av.open(args.input)
    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    if len(input_container.streams.audio) > 0:
        audio_input_stream = input_container.streams.audio[0]
    else:
        audio_input_stream = None
    if input_container.duration:
        container_duration = float(input_container.duration / av.time_base)
    else:
        container_duration = None

    total = int(VU.get_duration(video_input_stream, container_duration) * video_input_stream.guessed_rate)
    pbar = tqdm(desc="Segmentation", total=total, ncols=80)
    video_no = 0
    output_container = open_output_video(args.output, ext, video_no, video_input_stream, audio_input_stream,
                                         args=args)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]
    audio_base_pts = video_base_pts = None
    prev_pts = -1
    video_filter = create_video_filter(video_input_stream, args)
    for packet in input_container.demux(streams):
        if packet.stream.type == "video":
            for frame in packet.decode():
                frame = video_filter.update(frame)
                if frame is None:
                    continue
                if prev_pts in pts:
                    # next video
                    close_output_video(output_container, video_filter)
                    video_no += 1
                    output_container = open_output_video(args.output, ext, video_no, video_input_stream, audio_input_stream,
                                                         args=args)
                    audio_base_pts = video_base_pts = None
                    video_filter = create_video_filter(video_input_stream, args)

                prev_pts = frame.pts
                if video_base_pts is None:
                    video_base_pts = frame.pts
                if args.reset_timestamp:
                    frame.pts = frame.pts - video_base_pts
                    frame.dts = None

                enc_packet = output_container.streams.video[0].encode(frame)
                if enc_packet:
                    output_container.mux(enc_packet)
                pbar.update(1)
        elif packet.stream.type == "audio":
            if packet.dts is None:
                continue

            if audio_base_pts is None:
                audio_base_pts = packet.pts

            packet.stream = output_container.streams.audio[0]
            if args.reset_timestamp:
                packet.pts = packet.pts - audio_base_pts
                packet.dts = None
            output_container.mux(packet)

    close_output_video(output_container, video_filter)
    input_container.close()
    pbar.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids. -1 for CPU")
    parser.add_argument("--window-size", type=int, default=300, help="frame size for inference step")
    parser.add_argument("--padding-size", type=int, default=25, help="padding/overlap frame size")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability threshold")
    parser.add_argument("--reset-timestamp", action="store_true", help="reset timestamp")
    parser.add_argument("--crf", type=int, default=0, help="crf")
    parser.add_argument("--codec", type=str, default=VU.LIBH264, choices=["libx264", "libopenh264", "ffv1"])
    parser.add_argument("--max-height", type=int, help="max height px for downscaling")
    args = parser.parse_args()

    if args.max_height is not None and args.max_height % 2 != 0:
        raise ValueError("--max-height must be multiple of 2")

    os.makedirs(args.output, exist_ok=True)

    device = create_device(args.gpu)
    pts = SBD.detect_boundary(
        args.input,
        device=device,
        window_size=args.window_size,
        padding_size=args.padding_size,
        threshold=args.threshold,
        max_fps=None,  # raw pts
    )
    segment_video(pts, args)

    if device.type == "cuda":
        max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        logger.debug(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    main()
