from fractions import Fraction
from typing import List

import av

from .color_transform import InputTransform
from .metadata import VideoMetadata
from .video_filter.av_filter_graph import AVFilterGraph
from .video_filter.fps import FPSFilter
from .video_filter.tensor_filter_graph import TensorFilterGraph


class VideoPreprocessor:
    fps_filter: FPSFilter | None
    input_transform: InputTransform | None
    video_filter: TensorFilterGraph | AVFilterGraph | None

    def __init__(
        self,
        video_stream: av.VideoStream,
        sw_format: VideoMetadata,
        fps: Fraction | None = None,
        vf: str = "",
        deny_filters: List[str] = [],
        input_transform: InputTransform | None = None,
    ):
        self.fps_filter = None
        self.video_filter = None
        self.input_transform = input_transform

        if fps is not None:
            if video_stream.guessed_rate is None or video_stream.time_base is None:
                raise RuntimeError("guessed_rate/time_base is None")
            self.fps_filter = FPSFilter(fps, video_stream.time_base, video_stream.guessed_rate)
        if vf:
            if self.input_transform is not None:
                self.video_filter = TensorFilterGraph(vf, deny_filters=deny_filters)
            else:
                self.video_filter = AVFilterGraph(video_stream, sw_format, vf, deny_filters)

    def update(self, frame):
        if self.fps_filter is not None:
            frames = self.fps_filter.update(frame)
        else:
            frames = [frame]

        out_frames = []
        for frame in frames:
            if self.input_transform is not None:
                frame = self.input_transform(frame)

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
            if self.input_transform is not None:
                frame = self.input_transform(frame)

            if self.video_filter is not None:
                frame = self.video_filter.update(frame)
                if frame is not None:
                    out_frames.append(frame)
            else:
                out_frames.append(frame)

        if self.video_filter is not None:
            out_frames += self.video_filter.flush()

        return out_frames
