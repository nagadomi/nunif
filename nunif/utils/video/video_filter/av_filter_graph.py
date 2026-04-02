import av
import re
from typing import List, Optional, Tuple


class AVFilterGraph:
    graph: av.filter.Graph

    def __init__(
        self,
        video_stream: av.video.stream.VideoStream,
        vf: str,
        deny_filters: Optional[List[str]] = None,
    ):
        self.graph = av.filter.Graph()
        deny_filters = deny_filters or []
        video_filters = self.parse_vf_option(vf)
        video_filters = [
            (name, option) for name, option in video_filters if name not in deny_filters
        ]
        self.build_graph(self.graph, video_stream, video_filters)

    def update(self, frame: av.VideoFrame) -> Optional[av.VideoFrame]:
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
        template_stream: av.video.stream.VideoStream,
        video_filters: List[Tuple[str, str]],
    ) -> None:
        buffer = graph.add_buffer(template=template_stream)
        prev_filter = buffer
        for filter_name, filter_option in video_filters:
            new_filter = graph.add(filter_name, filter_option if filter_option else None)
            prev_filter.link_to(new_filter)
            prev_filter = new_filter
        buffersink = graph.add("buffersink")
        prev_filter.link_to(buffersink)
        graph.configure()
