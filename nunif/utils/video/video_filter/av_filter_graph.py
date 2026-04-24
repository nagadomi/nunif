import re
from fractions import Fraction
from typing import List, Tuple

import av

from ..metadata import HW_PIX_FORMATS, VideoMetadata


class AVFilterGraph:
    graph: av.filter.Graph

    def __init__(
        self,
        stream_pix_fmt: str,
        sw_format: VideoMetadata,
        vf: str,
        deny_filters: List[str] | None = None,
    ):
        self.pix_fmt = sw_format.guess_pix_fmt(stream_pix_fmt)
        self.sw_format = sw_format
        deny_filters = deny_filters or []
        video_filters = self.parse_vf_option(vf)
        video_filters = [(name, option) for name, option in video_filters if name not in deny_filters]
        self.graph = av.filter.Graph()
        self.graph.threads = 1
        self.build_graph(
            self.graph,
            pix_fmt=self.pix_fmt,
            width=sw_format.width,
            height=sw_format.height,
            time_base=sw_format.time_base,
            video_filters=video_filters,
        )

    def update(self, frame: av.VideoFrame) -> av.VideoFrame | None:
        if frame.format.name in HW_PIX_FORMATS:
            # hwdownload
            frame = frame.reformat(
                format=self.pix_fmt,
                src_colorspace=self.sw_format.colorspace,
                src_color_range=self.sw_format.color_range,
                dst_colorspace=self.sw_format.colorspace,
                dst_color_primaries=self.sw_format.color_primaries,
                dst_color_trc=self.sw_format.color_trc,
                dst_color_range=self.sw_format.color_range,
            )
        self.graph.push(frame)
        try:
            out_frame = self.graph.pull()
            if isinstance(out_frame, av.VideoFrame):
                return out_frame
        except av.error.BlockingIOError:
            return None
        except av.error.EOFError:
            return None
        return None

    def flush(self) -> List[av.VideoFrame]:
        try:
            self.graph.push(None)
        except av.error.EOFError:
            pass

        out_frames: List[av.VideoFrame] = []
        while True:
            try:
                out_frame = self.graph.pull()
                if isinstance(out_frame, av.VideoFrame):
                    out_frames.append(out_frame)
                else:
                    break
            except av.error.BlockingIOError:
                break
            except av.error.EOFError:
                break
        return out_frames

    @staticmethod
    def parse_vf_option(vf: str) -> List[Tuple[str, str]]:
        video_filters: List[Tuple[str, str]] = []
        vf = vf.strip()
        if not vf:
            return video_filters

        # split by ',' not preceded by '\'
        for line in re.split(r"(?<!\\),", vf):
            line = line.strip().replace(r"\,", ",")
            if line:
                # split by '=' not preceded by '\'
                col = re.split(r"(?<!\\)=", line, 1)
                if len(col) == 2:
                    filter_name, filter_option = col
                else:
                    filter_name, filter_option = col[0], ""

                filter_name = filter_name.strip().replace(r"\=", "=")
                filter_option = filter_option.strip().replace(r"\=", "=")
                video_filters.append((filter_name, filter_option))
        return video_filters

    @staticmethod
    def build_graph(
        graph: av.filter.Graph,
        pix_fmt: str,
        width: int,
        height: int,
        time_base: Fraction | None,
        video_filters: List[Tuple[str, str]],
    ) -> None:
        buffer = graph.add_buffer(
            format=av.VideoFormat(pix_fmt, width=width, height=height),
            width=width,
            height=height,
            time_base=time_base,
        )
        prev_filter = buffer
        for filter_name, filter_option in video_filters:
            new_filter = graph.add(filter_name, filter_option if filter_option else None)
            prev_filter.link_to(new_filter)
            prev_filter = new_filter
        buffersink = graph.add("buffersink")
        prev_filter.link_to(buffersink)
        graph.configure()
