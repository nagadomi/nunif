from fractions import Fraction
from typing import Any, Dict, List

import av
import torch

from ..color_lut import get_hdr2sdr_lut_path
from .color_transform import InputTransform, TensorFrame
from .hwaccel import should_use_tensor_frame
from .metadata import COLORSPACE_BT2020, ColorTrc, VideoMetadata
from .utils import is_discrete_device
from .video_filter.av_filter_graph import AVFilterGraph
from .video_filter.fps import FPSFilter
from .video_filter.tensor_filter_graph import TensorFilterGraph


def _get_lut_path(colorspace: int, color_trc: int, output_colorspace: str) -> str:
    return get_hdr2sdr_lut_path(
        {
            (ColorTrc.SMPTE2084, "bt709"): "pq2bt709",
            (ColorTrc.SMPTE2084, "bt601"): "pq2bt601",
            (ColorTrc.ARIB_STD_B67, "bt709"): "hlg2bt709",
            (ColorTrc.ARIB_STD_B67, "bt601"): "hlg2bt601",
        }[(ColorTrc(color_trc), output_colorspace.split("-")[0])]
    )


def _is_hdr2sdr_enabled(colorspace: int, color_trc: int, output_colorspace: str) -> bool:
    return (
        colorspace == COLORSPACE_BT2020
        and color_trc in {ColorTrc.SMPTE2084, ColorTrc.ARIB_STD_B67}
        and output_colorspace in {"bt709", "bt709-tv", "bt709-pc", "bt601", "bt601-tv", "bt601-pc"}
    )


def _update_hdr2sdr_video_filter(vf: str, colorspace: int, color_trc: int, output_colorspace: str) -> str:
    lut_path = _get_lut_path(colorspace=colorspace, color_trc=color_trc, output_colorspace=output_colorspace)
    if "bt709" in output_colorspace:
        colorspace_filter = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
    else:
        colorspace_filter = "setparams=colorspace=smpte170m:color_primaries=bt470bg:color_trc=smpte170m"

    lut_filter = f"lut3d={lut_path},{colorspace_filter}"
    vf = f"{vf},{lut_filter}" if vf else lut_filter
    return vf


class VideoPreprocessor:
    fps_filter: FPSFilter | None
    input_transform: InputTransform | None
    video_filter: TensorFilterGraph | AVFilterGraph | None
    reformatter: Any | None

    def __init__(
        self,
        stream_pix_fmt: str,
        sw_format: VideoMetadata,
        output_colorspace_mode: str = "auto",
        fps: Fraction | None = None,
        vf: str = "",
        deny_filters: List[str] = [],
        hwaccel: str | None = None,
        device: torch.device | None = None,
        input_reformat_options: Dict[str, Any] | None = None,
    ):
        self.fps_filter = None
        self.video_filter = None
        self.input_transform = None
        self.reformatter = None
        self.shared_reformatter = av.video.reformatter.VideoReformatter()

        if device is None:
            device = torch.device("cpu")

        use_hdr2sdr = _is_hdr2sdr_enabled(
            colorspace=sw_format.colorspace,
            color_trc=sw_format.color_trc,
            output_colorspace=output_colorspace_mode,
        )
        use_tensor_frame = should_use_tensor_frame(sw_format.format.name, hwaccel, device)

        if use_hdr2sdr:
            vf = _update_hdr2sdr_video_filter(
                vf,
                colorspace=sw_format.colorspace,
                color_trc=sw_format.color_trc,
                output_colorspace=output_colorspace_mode,
            )
            if use_tensor_frame:
                (
                    _,
                    colorspace,
                    color_primaries,
                    color_trc,
                    _,
                ) = sw_format.get_target_colorspace(colorspace_mode="auto", pix_fmt=sw_format.format.name)
                input_reformat_options = sw_format.get_input_reformat_options(colorspace, color_primaries, color_trc)

        if fps is not None:
            if sw_format.guessed_rate is None or sw_format.time_base is None:
                raise RuntimeError("guessed_rate/time_base is None")
            self.fps_filter = FPSFilter(fps, sw_format.time_base, sw_format.guessed_rate)

        if use_tensor_frame:
            assert input_reformat_options is not None
            self.input_transform = InputTransform(
                src_pix_fmt=sw_format.format.name,
                src_colorspace=input_reformat_options["src_colorspace"],
                src_color_primaries=input_reformat_options["src_color_primaries"],
                src_color_trc=input_reformat_options["src_color_trc"],
                src_color_range=input_reformat_options["src_color_range"],
                dst_colorspace=input_reformat_options["dst_colorspace"],
                dst_color_primaries=input_reformat_options["dst_color_primaries"],
                dst_color_trc=input_reformat_options["dst_color_trc"],
                dst_color_range=input_reformat_options["dst_color_range"],
                use_16bit=sw_format.use_16bit,
                device=device,
            )
            if vf:
                self.video_filter = TensorFilterGraph(vf, deny_filters=deny_filters)
        else:
            if vf:
                self.video_filter = AVFilterGraph(stream_pix_fmt, sw_format, vf, deny_filters)
            if input_reformat_options is not None:
                # Optimized transfer for software decoders when device is GPU
                dlpack_pix_fmt = sw_format.guess_sw_dlpack_pix_fmt()
                if dlpack_pix_fmt is not None and is_discrete_device(device):
                    dst_pix_fmt: str = dlpack_pix_fmt
                else:
                    dst_pix_fmt = sw_format.guess_rgb_pix_fmt()

                def _reformatter(frame: av.VideoFrame) -> av.VideoFrame:
                    dst_color_trc = (
                        None
                        if input_reformat_options["dst_color_trc"] == frame.color_trc
                        else input_reformat_options["dst_color_trc"]
                    )
                    dst_color_primaries = (
                        None
                        if input_reformat_options["dst_color_primaries"] == frame.color_primaries
                        else input_reformat_options["dst_color_primaries"]
                    )
                    new_frame = self.shared_reformatter.reformat(
                        frame,
                        format=dst_pix_fmt,
                        src_colorspace=input_reformat_options["src_colorspace"],
                        src_color_range=input_reformat_options["src_color_range"],
                        dst_colorspace=input_reformat_options["dst_colorspace"],
                        dst_color_primaries=dst_color_primaries,
                        dst_color_trc=dst_color_trc,
                        dst_color_range=input_reformat_options["dst_color_range"],
                    )
                    return new_frame

                self.reformatter = _reformatter

    def _apply_reformat(self, frame: av.VideoFrame) -> av.VideoFrame | TensorFrame:
        if self.input_transform is not None:
            # frame is av.VideoFrame -> TensorFrame
            return self.input_transform(frame)
        if self.reformatter is not None:
            # frame is av.VideoFrame -> av.VideoFrame (reformatted)
            return self.reformatter(frame)
        return frame

    def update(self, frame: av.VideoFrame) -> List[av.VideoFrame | TensorFrame]:
        if self.fps_filter is not None:
            frames = self.fps_filter.update(frame)
        else:
            frames = [frame]

        out_frames: List[av.VideoFrame | TensorFrame] = []
        for frame in frames:
            if self.video_filter is not None:
                if self.input_transform is not None:
                    t_frame: TensorFrame = self.input_transform(frame)
                    assert isinstance(self.video_filter, TensorFilterGraph)
                    res_t = self.video_filter.update(t_frame)
                    if res_t is not None:
                        out_frames.append(res_t)
                else:
                    assert isinstance(self.video_filter, AVFilterGraph)
                    res_v = self.video_filter.update(frame)
                    if res_v is not None:
                        out_frames.append(res_v)
            else:
                # No video filters, just apply reformat (tensor or software)
                out_frames.append(self._apply_reformat(frame))

        return out_frames

    def flush(self) -> List[av.VideoFrame | TensorFrame]:
        if self.fps_filter is not None:
            frames = self.fps_filter.flush()
        else:
            frames = []

        out_frames: List[av.VideoFrame | TensorFrame] = []
        for frame in frames:
            if self.video_filter is not None:
                if self.input_transform is not None:
                    t_frame: TensorFrame = self.input_transform(frame)
                    assert isinstance(self.video_filter, TensorFilterGraph)
                    res_t = self.video_filter.update(t_frame)
                    if res_t is not None:
                        out_frames.append(res_t)
                else:
                    assert isinstance(self.video_filter, AVFilterGraph)
                    res_v = self.video_filter.update(frame)
                    if res_v is not None:
                        out_frames.append(res_v)
            else:
                out_frames.append(self._apply_reformat(frame))

        if self.video_filter is not None:
            # Final output from filters
            final_frames: List[Any] = self.video_filter.flush()
            for f_final in final_frames:
                out_frames.append(f_final)

        return out_frames
