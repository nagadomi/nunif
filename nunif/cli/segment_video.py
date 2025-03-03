# Split video by Shot Boundary Detection (TransNetV2)
import nunif.utils.video as VU
import av
from nunif.utils.transnetv2 import TransNetV2
import torch
import argparse
import os
from os import path
from packaging import version as packaging_version
from nunif.device import create_device, autocast
from tqdm import tqdm


def fps_config_callback(args, fps_hook=None):
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
        with torch.inference_mode():  #, autocast(x.device):
            single_frame_pred, all_frame_pred = model(x.unsqueeze(0).permute(0, 1, 3, 4, 2))
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
                  config_callback=fps_config_callback(args),
                  title="Shot Boundary Detection",
                  vf="scale=48:27")

    last_x = frames[-1][0][-1:]
    last_pts = frames[-1][1][-1:]
    pad_x = torch.cat((last_x,) * args.padding_size, dim=0)
    pad_pts = torch.cat((last_pts,) * args.padding_size, dim=0)
    while results[-1][1][-1] != last_pts[0]:
        frames.append((pad_x, pad_pts))
        if len(frames) == args.window_size // args.padding_size:
            push_predict(torch.cat([x_ for x_, _ in frames], dim=0),
                         torch.cat([pts_ for _, pts_ in frames], dim=0))

    frame_preds = torch.cat([pred for pred, pts in results], dim=0)[:frame_count[0]]
    frame_pts = torch.cat([pts for pred, pts in results], dim=0)[:frame_count[0]]
    segment_pts = frame_pts[frame_preds > args.threshold]

    return segment_pts


AV_VERSION_14 = packaging_version.parse(av.__version__).major >= 14


def add_stream_from_template(container, template):
    # wrapper for av >= 14 compatibility
    if AV_VERSION_14:
        return container.add_stream_from_template(template)
    else:
        return container.add_stream(template=template)


def open_output_video(output_path, ext, no, video_input_stream, audio_input_stream):
    output_container = av.open(path.join(output_path, f"{str(no).zfill(6)}{ext}"), "w")
    video_output_stream = add_stream_from_template(output_container, template=video_input_stream)
    video_output_stream.thread_type = "AUTO"

    if audio_input_stream is not None:
        add_stream_from_template(output_container, template=audio_input_stream)

    return output_container


def segment_video(pts, args):
    ext = path.splitext(path.basename(args.input))[-1]
    input_container = av.open(args.input)
    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    if len(input_container.streams.audio) > 0:
        # has audio stream
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
    output_container = open_output_video(args.output, ext, video_no, video_input_stream, audio_input_stream)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]
    for packet in input_container.demux(streams):
        if packet.dts is None:
            continue

        if packet.stream.type == "video":
            packet.stream = output_container.streams.video[0]
            output_container.mux(packet)

            if packet.pts in pts:
                # next video
                output_container.close()
                video_no += 1
                output_container = open_output_video(args.output, ext, video_no, video_input_stream, audio_input_stream)

            pbar.update(int(packet.pts * video_input_stream.time_base * video_input_stream.guessed_rate) - pbar.n)
        elif packet.stream.type == "audio":
            packet.stream = output_container.streams.audio[0]
            output_container.mux(packet)

    output_container.close()
    input_container.close()
    pbar.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="gpu ids. -1 for CPU")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--padding-size", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    assert (args.window_size % args.padding_size == 0 and
            args.window_size // args.padding_size >= 3)  # pad1 + frames + pad2

    device = create_device(args.gpu)
    model = TransNetV2().load().eval().to(device)
    pts = detect_segments(model, device, args)
    pts = pts.tolist()
    segment_video(pts, args)


if __name__ == "__main__":
    main()
