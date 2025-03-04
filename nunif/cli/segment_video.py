# Split video by Shot Boundary Detection (TransNetV2)
# python -m nunif.cli.semgent_video -i input.mp4 -o output_dir
import nunif.utils.video as VU
import av
from nunif.utils.transnetv2 import TransNetV2
import torch
import argparse
import os
from os import path
from tqdm import tqdm
from nunif.device import create_device
from nunif.logger import logger


def source_fps_config(args, fps_hook=None):
    """ keep original frame rate
    """
    def callback(stream):
        return VU.VideoOutputConfig(
            fps=None,
        )
    return callback


def detect_segments(model, device, args):
    frames = []
    results = []
    first_frame = [True]
    frame_count = [0]

    def push_predict(x, pts):
        # NOTE: No need autocast. Nothing improves.
        with torch.inference_mode():
            single_frame_pred, all_frame_pred = model(x)
            single_frame_pred = torch.sigmoid(single_frame_pred).flatten()
        results.append((
            single_frame_pred[args.padding_size:-args.padding_size].cpu(),
            pts[args.padding_size:-args.padding_size].cpu()
        ))
        for _ in range((args.window_size - args.padding_size * 2) // args.padding_size):
            frames.pop(0)

    def batch_callback(x, pts):
        frame_count[0] += x.shape[0]
        pts = torch.tensor(pts, dtype=torch.long)
        if x.shape[0] < args.padding_size:
            n = args.padding_size - x.shape[0]
            pad_x = torch.cat((x[-1:],) * n, dim=0)
            pad_pts = torch.cat((pts[-1:],) * n, dim=0)
            x = torch.cat((x, pad_x), dim=0)
            pts = torch.cat((pts, pad_pts), dim=0)

        if first_frame[0]:
            first_frame[0] = False
            pad_x = torch.cat((x[0:1],) * args.padding_size, dim=0)
            pad_pts = torch.cat((pts[0:1],) * args.padding_size, dim=0)
            frames.append((pad_x, pad_pts))
            frames.append((x, pts))
        else:
            frames.append((x, pts))

        if len(frames) == args.window_size // args.padding_size:
            push_predict(torch.cat([x_ for x_, _ in frames], dim=0),
                         torch.cat([pts_ for _, pts_ in frames], dim=0))

    callback_pool = VU.FrameCallbackPool(
        batch_callback,
        require_pts=True,
        batch_size=args.padding_size,
        device=device,
        max_workers=0,  # must be sequential
    )
    VU.hook_frame(args.input, callback_pool,
                  config_callback=source_fps_config(args),
                  title="Shot Boundary Detection",
                  vf="scale=48:27")  # input size for TransNetV2

    last_x = frames[-1][0][-1:]
    last_pts = frames[-1][1][-1:]
    pad_x = torch.cat((last_x,) * args.padding_size, dim=0)
    pad_pts = torch.cat((last_pts,) * args.padding_size, dim=0)
    while (not results or results[-1][1][-1] != last_pts[0]):
        frames.append((pad_x, pad_pts))
        if len(frames) == args.window_size // args.padding_size:
            push_predict(torch.cat([x_ for x_, _ in frames], dim=0),
                         torch.cat([pts_ for _, pts_ in frames], dim=0))

    frame_preds = torch.cat([pred for pred, pts in results], dim=0)[:frame_count[0]]
    frame_pts = torch.cat([pts for pred, pts in results], dim=0)[:frame_count[0]]
    segment_pts = frame_pts[frame_preds > args.threshold]

    return segment_pts


def open_output_video(output_path, ext, no, video_input_stream, audio_input_stream, args):
    output_container = av.open(path.join(output_path, f"{str(no).zfill(6)}{ext}"), "w")
    video_output_stream = output_container.add_stream(args.codec, rate=video_input_stream.guessed_rate)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = video_input_stream.pix_fmt
    video_output_stream.width = video_input_stream.width
    video_output_stream.height = video_input_stream.height

    if args.codec == "libx264":
        video_output_stream.options = {"crf": str(args.crf), "preset": "veryfast"}
    elif args.codec == "libopenh264":
        video_output_stream.options = {"b": "24M"}
    video_output_stream.thread_type = "AUTO"
    if audio_input_stream is not None:
        # TODO: Some formats cause errors
        VU.add_stream_from_template(output_container, template=audio_input_stream)

    return output_container


def close_output_video(output_container):
    enc_packet = output_container.streams.video[0].encode(None)
    if enc_packet:
        output_container.mux(enc_packet)
    output_container.close()


def segment_video(pts, args):
    if args.codec == "ffv1":
        ext = ".mkv"
    elif args.codec == "utvideo":
        ext = ".avi"
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
    for packet in input_container.demux(streams):
        if packet.stream.type == "video":
            for frame in packet.decode():
                if prev_pts in pts:
                    # next video
                    close_output_video(output_container)
                    video_no += 1
                    output_container = open_output_video(args.output, ext, video_no, video_input_stream, audio_input_stream,
                                                         args=args)
                    audio_base_pts = video_base_pts = None

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

    close_output_video(output_container)
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
    parser.add_argument("--codec", type=str, default=VU.LIBH264, choices=["libx264", "libopenh264", "ffv1", "utvideo"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    assert (args.window_size % args.padding_size == 0 and
            args.window_size // args.padding_size >= 3)  # pad1 + frames + pad2

    device = create_device(args.gpu)
    model = TransNetV2().load().eval().to(device)
    pts = detect_segments(model, device, args)
    pts = pts.tolist()
    segment_video(pts, args)

    if device.type == "cuda":
        max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        logger.debug(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    main()
