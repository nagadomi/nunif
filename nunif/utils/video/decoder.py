import sys
from typing import Literal, cast

import av
import av.codec.context
import av.video.codeccontext
import av.video.stream

from .utils import has_vp9

SkipFrameType = Literal["NONE", "DEFAULT", "NONREF", "BIDIR", "NONINTRA", "NONKEY", "ALL"]


def is_vp9_alpha(stream: av.video.stream.VideoStream) -> bool:
    if stream.codec_context.name != "vp9":
        return False
    alpha_mode = stream.metadata.get("alpha_mode")
    if not alpha_mode and stream.container is not None:
        alpha_mode = stream.container.metadata.get("alpha_mode")
    return alpha_mode in {"1", "true", "True"}


class VideoDecoder:
    def __init__(
        self,
        stream: av.video.stream.VideoStream,
        strict: bool = False,
    ):
        self.stream = stream
        self.strict = strict
        self.custom_ctx: av.video.codeccontext.VideoCodecContext | None = None

        if is_vp9_alpha(stream) and has_vp9():
            ctx = cast(av.video.codeccontext.VideoCodecContext, av.codec.CodecContext.create("libvpx-vp9", "r"))
            ctx.thread_type = "AUTO"
            self.custom_ctx = ctx

    @property
    def skip_frame(self) -> SkipFrameType:
        if self.custom_ctx is not None:
            return self.custom_ctx.skip_frame
        return self.stream.codec_context.skip_frame

    @skip_frame.setter
    def skip_frame(self, value: SkipFrameType) -> None:
        self.stream.codec_context.skip_frame = value
        if self.custom_ctx is not None:
            self.custom_ctx.skip_frame = value

    def flush_buffers(self) -> None:
        if self.custom_ctx is not None:
            self.custom_ctx.flush_buffers()
        self.stream.codec_context.flush_buffers()

    def decode(self, packet: av.Packet) -> list[av.VideoFrame]:
        if self.custom_ctx is not None:
            return self._decode(self.custom_ctx, packet)
        return self._decode(packet)

    def _decode(
        self,
        target: av.Packet | av.video.codeccontext.VideoCodecContext,
        packet: av.Packet | None = None,
    ) -> list[av.VideoFrame]:
        try:
            if isinstance(target, av.video.codeccontext.VideoCodecContext):
                return target.decode(packet)
            return target.decode()
        except av.error.EOFError:
            return []
        except av.error.InvalidDataError:
            if self.strict:
                raise
            print("\n[WARN] Input video has invalid data/frames! continuing anyway...", file=sys.stderr)
            return []
        except av.error.PermissionError:
            if self.strict:
                raise
            print("\n[WARN] No permission to read data/frames! continuing anyway...", file=sys.stderr)
            return []
        except av.error.PatchWelcomeError:
            if self.strict:
                raise
            print("\n[WARN] Unknown data/frames type (pyAV Unimplemented)! continuing anyway...", file=sys.stderr)
            return []
