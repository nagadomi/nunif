from .video_filter.fps import FPSFilter
from .av_filter_graph import AVFilterGraph


class VideoPreprocessor():
    def __init__(self, video_stream, fps=None, vf="", deny_filters=[]):
        self.fps_filter = None
        self.video_filter = None

        if fps is not None:
            self.fps_filter = FPSFilter(fps, video_stream.time_base, video_stream.guessed_rate)
        if vf:
            self.video_filter = AVFilterGraph(video_stream, vf, deny_filters)

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
