import av


def is_vaapi_codec(codec) -> bool:
    return isinstance(codec, str) and codec.endswith("_vaapi")


class VaapiUploader:
    """Filter graph that uploads system-memory frames to VAAPI surfaces.

    The graph is:
        [buffer] -> [hwupload] -> [format=vaapi] -> [buffersink]

    The hwupload filter's `derive_device=vaapi` option tells it to pick up the
    VAAPI device from the downstream `format=vaapi` filter, which in turn reads
    the device path from the output stream's `vaapi_device` option.

    The graph is built lazily on the first frame because the output size is
    not known until the first frame is reformatted.
    """

    def __init__(self, time_base=None):
        self.time_base = time_base
        self.graph = None

    def _build_graph(self, frame):
        self.graph = av.filter.Graph()
        buffer = self.graph.add_buffer(
            width=frame.width,
            height=frame.height,
            format=frame.format.name,
            time_base=self.time_base or frame.time_base,
        )
        buffer.link_to(self.graph.add("hwupload", "derive_device=vaapi"))
        self.graph.add("format", "vaapi").link_to(self.graph.add("buffersink"))
        self.graph.configure()

    def update(self, frame):
        if frame is None:
            return None
        if self.graph is None:
            self._build_graph(frame)
        self.graph.push(frame)
        try:
            return self.graph.pull()
        except av.error.BlockingIOError:
            return None
        except av.error.EOFError:
            return None
