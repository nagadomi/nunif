import av
import math
from tqdm import tqdm


def get_fps(stream):
    return stream.guessed_rate


def gusess_frames(stream):
    return math.ceil(float(stream.duration * stream.time_base) * get_fps(stream))


def get_duration(stream):
    return math.ceil(float(stream.duration * stream.time_base))


def get_frames(stream):
    if stream.frames > 0:
        return stream.frames
    else:
        # frames is unknown
        return gusess_frames(stream)


def _print_len(stream):
    print("frames", stream.frames)
    print("guessed_frames", gusess_frames(stream))
    print("duration", get_duration(stream))
    print("base_rate", float(stream.base_rate))
    print("average_rate", float(stream.average_rate))
    print("guessed_rate", float(stream.guessed_rate))


class FixedFPSFilter():
    def __init__(self, video_stream, fps):
        self.graph = av.filter.Graph()
        buffer = self.graph.add_buffer(template=video_stream)
        fps_filter = self.graph.add("fps", str(fps))
        buffersink = self.graph.add("buffersink")
        buffer.link_to(fps_filter)
        fps_filter.link_to(buffersink)
        self.graph.configure()

    def update(self, frame):
        self.graph.push(frame)
        try:
            return self.graph.pull()
        except av.error.BlockingIOError:
            return None
        except av.error.EOFError:
            # finished
            return None


class VideoOutputConfig():
    def __init__(self, width=None, height=None, pix_fmt="yuv420p", fps=30, options={}):
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.options = options


def default_config_callback(stream):
    fps = get_fps(stream)
    if float(fps) > 30:
        fps = 30
    return VideoOutputConfig(
        stream.codec_context.width, stream.codec_context.height,
        fps=fps,
        options={"preset": "ultrafast", "crf": "20"}
    )


def process_video(input_path, output_path,
                  frame_callback,
                  config_callback=default_config_callback,
                  title=None):
    input_container = av.open(input_path)
    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    # _print_len(video_input_stream)
    audio_input_stream = audio_output_stream = None
    if len(input_container.streams.audio) > 0:
        # has audio stream
        audio_input_stream = input_container.streams.audio[0]

    config = config_callback(video_input_stream)
    output_container = av.open(output_path, 'w')

    fps_filter = FixedFPSFilter(video_input_stream, config.fps)
    video_output_stream = output_container.add_stream("libx264", config.fps)
    video_output_stream.thread_type = "AUTO"
    video_output_stream.pix_fmt = config.pix_fmt
    video_output_stream.width = config.width
    video_output_stream.height = config.height
    video_output_stream.options = config.options
    if audio_input_stream is not None:
        try:
            audio_output_stream = output_container.add_stream(template=audio_input_stream)
            audio_copy = True
        except ValueError:
            audio_output_stream = output_container.add_stream("aac", audio_input_stream.rate)
            audio_copy = False

    pbar = tqdm(desc=(title if title else output_path), total=get_frames(video_input_stream), ncols=80)
    streams = [s for s in [video_input_stream, audio_input_stream] if s is not None]
    for packet in input_container.demux(streams):
        if packet.stream.type == "video":
            for frame in packet.decode():
                frame = fps_filter.update(frame)
                if frame is not None:
                    new_frame = frame_callback(frame)
                    enc_packet = video_output_stream.encode(new_frame)
                    if enc_packet:
                        output_container.mux(enc_packet)
                pbar.update(1)

        elif packet.stream.type == "audio":
            if packet.dts is not None:
                if audio_copy:
                    packet.stream = audio_output_stream
                else:
                    for frame in packet.decode():
                        frame.pts = None
                        enc_packet = audio_output_stream.encode(frame)
                        if enc_packet:
                            output_container.mux(enc_packet)

    pbar.close()
    frame = fps_filter.update(None)
    if frame is not None:
        new_frame = frame_callback(frame)
        enc_packet = video_output_stream.encode(new_frame)
        if enc_packet:
            output_container.mux(enc_packet)
    packet = video_output_stream.encode(None)
    if packet:
        output_container.mux(packet)
    output_container.close()
    input_container.close()


if __name__ == "__main__":
    from PIL import Image, ImageOps
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
            stream.codec_context.width * 2, stream.codec_context.height,
            fps=fps,
            options={"preset": "ultrafast", "crf": "20"}
        )

    def process_image(frame):
        im = frame.to_image()
        mirror = ImageOps.mirror(im)
        new_im = Image.new("RGB", (im.width * 2, im.height))
        new_im.paste(im, (0, 0))
        new_im.paste(mirror, (im.width, 0))
        new_frame = frame.from_image(new_im)
        return new_frame

    process_video(args.input, args.output, config_callback=make_config, frame_callback=process_image)
