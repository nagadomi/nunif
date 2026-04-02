from fractions import Fraction
from typing import List, Optional
import av


class FPSFilter:
    """
    FPS filter Python implementation with behavior equivalent to the ffmpeg fps filter.
    Note that this overwrites the frame.pts and frame.time_base of the input frame.
    """

    target_fps: Fraction
    input_time_base: Fraction
    output_time_base: Fraction
    default_duration: int
    frames_out: Optional[int]
    last_frame: Optional[av.VideoFrame]

    def __init__(self, fps: Fraction, stream_time_base: Fraction, stream_fps: Fraction):
        assert isinstance(fps, Fraction)
        assert isinstance(stream_fps, Fraction)
        self.target_fps = fps
        self.input_time_base = stream_time_base
        self.output_time_base = 1 / self.target_fps
        self.default_duration = int(1 / (stream_fps * self.input_time_base))
        self.frames_out = None
        self.last_frame = None

    def _create_out_frame(self, src_frame: av.VideoFrame, pts: int) -> av.VideoFrame:
        # Keep original DTS if we want to match pyav's potentially inconsistent behavior
        # or set to None if that's what pyav does.
        # src_frame.dts = src_frame.dts
        src_frame.pts = pts
        src_frame.time_base = self.output_time_base
        return src_frame

    def update(self, frame: av.VideoFrame) -> List[av.VideoFrame]:
        # Assume frame.pts is never None here
        assert frame.pts is not None
        time_seconds = frame.pts * self.input_time_base
        expected_total_out = int(time_seconds * self.target_fps + Fraction(1, 2))

        if self.frames_out is None:
            self.frames_out = expected_total_out

        out_frames: List[av.VideoFrame] = []
        if self.last_frame is not None:
            nb_frames = expected_total_out - self.frames_out
            for _ in range(nb_frames):
                out_frames.append(
                    self._create_out_frame(self.last_frame, self.frames_out)
                )
                self.frames_out += 1

        self.last_frame = frame
        return out_frames

    def flush(self) -> List[av.VideoFrame]:
        if self.last_frame is None:
            return []

        assert self.last_frame.pts is not None
        duration = (
            self.last_frame.duration
            if self.last_frame.duration
            else self.default_duration
        )
        time_seconds = (self.last_frame.pts + duration) * self.input_time_base
        expected_total_out = int(time_seconds * self.target_fps + Fraction(1, 2))

        out_frames: List[av.VideoFrame] = []
        if self.frames_out is not None:
            nb_frames = expected_total_out - self.frames_out
            for _ in range(nb_frames):
                out_frames.append(
                    self._create_out_frame(self.last_frame, self.frames_out)
                )
                self.frames_out += 1

        return out_frames
