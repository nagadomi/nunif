import av
import math
from tqdm import tqdm


def get_fps(stream):
    return stream.guessed_rate


def guess_frames(stream, fps=None):
    fps = fps or get_fps(stream)
    return math.ceil(float(stream.duration * stream.time_base) * fps)


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
    print("guessed_frames", guess_frames(stream))
    print("duration", get_duration(stream))
    print("base_rate", float(stream.base_rate))
    print("average_rate", float(stream.average_rate))
    print("guessed_rate", float(stream.guessed_rate))


class FixedFPSFilter():
    @staticmethod
    def parse_vf_option(vf):
        video_filters = []
        vf = vf.strip()
        if not vf:
            return video_filters
        for line in vf.split(","):
            line = line.strip()
            if line:
                col = line.split("=", 1)
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

    def __init__(self, video_stream, fps, vf=""):
        self.graph = av.filter.Graph()
        video_filters = self.parse_vf_option(vf)
        video_filters.append(("fps", str(fps)))
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
                  title=None,
                  vf=""):
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

    fps_filter = FixedFPSFilter(video_input_stream, config.fps, vf)
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

    desc = (title if title else output_path)
    ncols = len(desc) + 60
    pbar = tqdm(desc=desc, total=guess_frames(video_input_stream, config.fps), ncols=ncols)
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
                    output_container.mux(packet)
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


def process_video_keyframes(input_path, frame_callback, min_interval_sec=4., title=None):
    input_container = av.open(input_path)
    if len(input_container.streams.video) == 0:
        raise ValueError("No video stream")

    video_input_stream = input_container.streams.video[0]
    video_input_stream.thread_type = "AUTO"
    video_input_stream.codec_context.skip_frame = "NONKEY"

    max_progress = get_duration(video_input_stream)
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
    pbar.update(max_progress - prev_sec)
    pbar.close()
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
