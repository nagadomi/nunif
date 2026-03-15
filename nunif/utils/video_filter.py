import av
from fractions import Fraction
import re


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


class FPSFilter():
    """
    FPS filter Python implementation with behavior equivalent to the ffmpeg fps filter.
    Note that this overwrites the frame.pts and frame.time_base of the input frame.
    """
    def __init__(self, fps, stream_time_base, stream_fps):
        fps = convert_fps_fraction(fps)
        stream_fps = convert_fps_fraction(stream_fps)
        stream_time_base = stream_time_base
        self.target_fps = fps
        self.input_time_base = stream_time_base
        self.output_time_base = 1 / self.target_fps
        self.default_duration = int(1 / (stream_fps * self.input_time_base))
        self.frames_out = None
        self.last_frame = None

    def _create_out_frame(self, src_frame, pts):
        src_frame.pts = pts
        # Keep original DTS if we want to match pyav's potentially inconsistent behavior
        # or set to None if that's what pyav does.
        # src_frame.dts = src_frame.dts
        src_frame.time_base = self.output_time_base
        return src_frame

    def update(self, frame):
        time_seconds = frame.pts * self.input_time_base
        expected_total_out = int(time_seconds * self.target_fps + Fraction(1, 2))

        if self.frames_out is None:
            self.frames_out = expected_total_out

        out_frames = []
        if self.last_frame is not None:
            nb_frames = expected_total_out - self.frames_out
            for _ in range(nb_frames):
                out_frames.append(
                    self._create_out_frame(self.last_frame, self.frames_out)
                )
                self.frames_out += 1

        self.last_frame = frame
        return out_frames

    def flush(self):
        if self.last_frame is None:
            return []

        duration = self.last_frame.duration if self.last_frame.duration else self.default_duration
        time_seconds = (self.last_frame.pts + duration) * self.input_time_base
        expected_total_out = int(time_seconds * self.target_fps + Fraction(1, 2))

        out_frames = []
        nb_frames = expected_total_out - self.frames_out
        for _ in range(nb_frames):
            out_frames.append(self._create_out_frame(self.last_frame, self.frames_out))
            self.frames_out += 1

        return out_frames


class VideoFilterGraph():
    def __init__(self, video_stream, vf, deny_filters=[]):
        self.graph = av.filter.Graph()
        video_filters = self.parse_vf_option(vf)
        video_filters = [(name, option) for name, option in video_filters if name not in deny_filters]
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

    def flush(self):
        out_frames = []
        while True:
            frame = self.update(None)
            if frame is not None:
                out_frames.append(frame)
            else:
                break
        return out_frames

    @staticmethod
    def parse_vf_option(vf):
        video_filters = []
        vf = vf.strip()
        if not vf:
            return video_filters

        for line in re.split(r'(?<!\\),', vf):
            line = line.strip()
            if line:
                col = re.split(r'(?<!\\)=', line, 1)
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


class VideoPreprocessor():
    def __init__(self, video_stream, fps=None, vf="", deny_filters=[]):
        self.fps_filter = None
        self.video_filter = None

        if fps is not None:
            self.fps_filter = FPSFilter(fps, video_stream.time_base, video_stream.guessed_rate)
        if vf:
            self.video_filter = VideoFilterGraph(video_stream, vf, deny_filters)

    def update(self, frame):
        if self.fps_filter is not None:
            frames = self.fps_filter.update(frame)
        else:
            frames = [frame]

        out_frames = []
        for frame in frames:
            if self.video_filter is not None:
                frame = self.video_filter.update(frame)
                if frame is not None:
                    out_frames.append(frame)
            else:
                out_frames.append(frame)

        return out_frames

    def flush(self):
        if self.fps_filter is not None:
            frames = self.fps_filter.flush()
        else:
            frames = []

        out_frames = []
        for frame in frames:
            if self.video_filter is not None:
                frame = self.video_filter.update(frame)
                if frame is not None:
                    out_frames.append(frame)
            else:
                out_frames.append(frame)

        if self.video_filter is not None:
            out_frames += self.video_filter.flush()

        return out_frames
