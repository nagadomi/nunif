from fractions import Fraction
from typing import Any, Dict, List, Set, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.dlpack
from av.codec.hwaccel import HWDeviceType
from av.video.frame import CudaContext
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

from .metadata import (
    COLORSPACE_UNSPECIFIED,
    VideoMetadata,
    get_rgb_pix_fmt,
)
from .utils import is_nvidia_gpu


class ColorTransform:
    # Normalized coefficients (Kr, Kb)
    COEFFS: Dict[str, Tuple[float, float]] = {
        "bt709": (0.2126, 0.0722),
        "bt601": (0.299, 0.114),
        "bt2020": (0.2627, 0.0593),
    }

    # Matrix cache
    _MATRIX_CACHE: Dict[Tuple[Any, ...], torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = {}

    # Format classification
    YUV_PLANAR: Set[str] = {
        "yuv420p",
        "yuv422p",
        "yuv444p",
        "yuvj420p",
        "yuvj422p",
        "yuvj444p",
        "yuv420p10le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "yuv420p16le",
        "yuv422p16le",
        "yuv444p16le",
    }
    YUV_SEMI_PLANAR: Set[str] = {"nv12", "p010le"}
    RGB_PLANAR: Set[str] = {
        "gbrp",
        "gbrap",
        "gbrp10le",
        "gbrp12le",
        "gbrp14le",
        "gbrp16le",
    }

    # packed formats: (bytes_per_pixel, channel_order)
    RGB_PACKED: Dict[str, Tuple[int, List[int]]] = {
        "rgb24": (3, [0, 1, 2]),
        "bgr24": (3, [2, 1, 0]),
        "rgba": (4, [0, 1, 2, 3]),
        "bgra": (4, [2, 1, 0, 3]),
        "rgb0": (4, [0, 1, 2]),
        "bgr0": (4, [2, 1, 0]),
    }

    # Formats with full color range by default
    FULL_RANGE_FORMATS: Set[str] = {"yuvj420p", "yuvj422p", "yuvj444p"}

    # Bit depth categorization
    BIT10_FORMATS: Set[str] = {
        "p010le",
        "yuv420p10le",
        "yuv422p10le",
        "yuv444p10le",
        "gbrp10le",
        "gbrap10le",
    }
    BIT12_FORMATS: Set[str] = {
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "gbrp12le",
        "gbrap12le",
    }
    BIT14_FORMATS: Set[str] = {"gbrp14le", "gbrap14le"}
    BIT16_FORMATS: Set[str] = {
        "yuv420p16le",
        "yuv422p16le",
        "yuv444p16le",
        "gbrp16le",
        "gbrap16le",
        "p016le",
    }

    # Formats supported by PyAV VideoPlane.__dlpack__
    DLPACK_SUPPORTED_FORMATS: Set[str] = {"cuda"}

    # Formats verified by testing
    VERIFIED_FORMATS: Set[str] = {"yuv420p", "yuv444p", "yuv420p10le"}

    @staticmethod
    @torch.no_grad()
    def to_tensor(
        frame: av.VideoFrame,
        dst_colorspace: Colorspace | int | None = None,
        dst_color_range: ColorRange | int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        upsample_mode: str = "bilinear",
    ) -> torch.Tensor:
        """Convert PyAV VideoFrame to CHW RGB tensor."""
        src_fmt: str = frame.format.name
        fmt: str = src_fmt
        planes_f: List[torch.Tensor] = []
        bpp: int = ColorTransform.RGB_PACKED[fmt][0] if fmt in ColorTransform.RGB_PACKED else 1

        for i, plane in enumerate(frame.planes):
            t: torch.Tensor | None = None
            if src_fmt in ColorTransform.DLPACK_SUPPORTED_FORMATS and hasattr(plane, "__dlpack__"):
                if i == 0:
                    # Ensure hardware decoder has finished writing to the planes.
                    if torch.cuda.default_stream() != torch.cuda.current_stream():
                        # hwcontext_cuda uses default stream(0x0)
                        torch.cuda.default_stream().synchronize()

                t = torch.from_dlpack(plane.__dlpack__())

                # TODO: Use sw_formt
                if t.dtype == torch.uint8:
                    fmt = "nv12"
                elif t.dtype == torch.uint16:
                    fmt = "p010le"
            else:
                stride: int = plane.line_size
                is_16bit: bool = (
                    fmt in ColorTransform.BIT10_FORMATS
                    or fmt in ColorTransform.BIT12_FORMATS
                    or fmt in ColorTransform.BIT14_FORMATS
                    or fmt in ColorTransform.BIT16_FORMATS
                )
                np_dtype: Any = np.uint16 if is_16bit else np.uint8
                itemsize: int = np.dtype(np_dtype).itemsize
                try:
                    actual_width: int = (
                        frame.width * bpp if i == 0 and fmt in ColorTransform.RGB_PACKED else plane.width
                    )
                    arr = np.frombuffer(plane, dtype=np_dtype).reshape(-1, stride // itemsize)[
                        : plane.height, :actual_width
                    ]
                    t = torch.from_numpy(arr.copy())
                except ValueError as e:
                    if "hardware frame" in str(e) or "CUDA" in str(e):
                        raise RuntimeError(
                            f"DLPack failed and fallback to NumPy is not possible for hardware frame ({fmt})"
                        ) from e
                    raise e

            assert t is not None
            if str(t.device) != str(device):
                t = t.to(device)

            # Optimization: Convert to dtype immediately and don't keep the original (DLPack) tensor.
            # This helps to free the underlying VideoFrame/hardware surface as soon as possible.
            planes_f.append(t.to(dtype))
            del t

        src_colorspace: Colorspace | int = frame.colorspace
        src_color_range: ColorRange | int = frame.color_range
        if fmt in ColorTransform.FULL_RANGE_FORMATS:
            src_color_range = ColorRange.JPEG

        div: float
        # Determine bit-depth based on fmt
        if fmt in ColorTransform.BIT10_FORMATS:
            div = 1023.0
            if fmt == "p010le":
                div = 65472.0  # 1023 << 6
        elif fmt in ColorTransform.BIT12_FORMATS:
            div = 4095.0
            if fmt == "p012le":
                div = 65520.0  # 4095 << 4
        elif fmt in ColorTransform.BIT14_FORMATS:
            div = 16383.0
        elif fmt in ColorTransform.BIT16_FORMATS:
            div = 65535.0
        else:
            div = 255.0

        rgb: torch.Tensor
        if fmt in ColorTransform.YUV_PLANAR:
            y, u, v = [p.unsqueeze(0).unsqueeze(0) for p in planes_f[:3]]
            rgb = ColorTransform.yuv_to_rgb(
                y,
                u,
                v,
                colorspace=src_colorspace,
                color_range=src_color_range,
                div=div,
                mode=upsample_mode,
            )
        elif fmt in ColorTransform.YUV_SEMI_PLANAR:
            y = planes_f[0].unsqueeze(0).unsqueeze(0)
            uv = planes_f[1]
            if uv.dim() == 2:
                uv = uv.reshape(uv.shape[0], uv.shape[1] // 2, 2)
            u = uv[:, :, 0].unsqueeze(0).unsqueeze(0)
            v = uv[:, :, 1].unsqueeze(0).unsqueeze(0)
            rgb = ColorTransform.yuv_to_rgb(
                y,
                u,
                v,
                colorspace=src_colorspace,
                color_range=src_color_range,
                div=div,
                mode=upsample_mode,
            )
        elif fmt in ColorTransform.RGB_PLANAR:
            g, b, r = [p.unsqueeze(0).unsqueeze(0).div_(div) for p in planes_f[:3]]
            rgb = torch.cat([r, g, b], dim=1)
        elif fmt in ColorTransform.RGB_PACKED:
            _, order = ColorTransform.RGB_PACKED[fmt]
            p0 = planes_f[0]
            if p0.dim() == 2:
                p0 = p0.reshape(frame.height, frame.width, bpp)
            channels = [p0[:, :, idx].unsqueeze(0).unsqueeze(0).div_(div) for idx in order]
            rgb = torch.cat(channels, dim=1)
        else:
            raise ValueError(f"Unsupported format for to_tensor: {fmt}")

        if dst_colorspace is not None and dst_color_range is not None:
            # Skip the roundtrip if the destination exactly matches the source.
            if dst_colorspace == src_colorspace and dst_color_range == src_color_range:
                pass
            elif fmt in ColorTransform.YUV_PLANAR or fmt in ColorTransform.YUV_SEMI_PLANAR:
                # Roundtrip for YUV sources is a no-op without chroma subsampling.
                pass
            else:
                y_rt, u_rt, v_rt = ColorTransform.rgb_to_yuv(
                    rgb,
                    colorspace=dst_colorspace,
                    color_range=dst_color_range,
                    out_format=fmt,
                )
                rgb = ColorTransform.yuv_to_rgb(
                    y_rt,
                    u_rt,
                    v_rt,
                    colorspace=dst_colorspace,
                    color_range=dst_color_range,
                    div=1.0,  # rgb_to_yuv returns [0, 1] normalized planes
                    mode=upsample_mode,
                )

        return rgb

    @staticmethod
    def get_coeffs(colorspace: Colorspace | int) -> Tuple[float, float]:
        """Return Kr, Kb coefficients based on colorspace numeric value."""
        value: int = int(colorspace)
        if value == 1:
            return ColorTransform.COEFFS["bt709"]
        elif value in (4, 5, 6, 7):
            return ColorTransform.COEFFS["bt601"]
        elif value in (9, 10):
            return ColorTransform.COEFFS["bt2020"]
        return ColorTransform.COEFFS["bt709"]

    @staticmethod
    def _get_matrix(
        colorspace: Colorspace | int,
        device: torch.device,
        dtype: torch.dtype,
        direction: str = "yuv2rgb",
        color_range: ColorRange | int | None = None,
        div: float = 255.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        key = (
            direction,
            int(colorspace),
            int(color_range) if color_range is not None else None,
            str(device),
            dtype,
            div,
        )
        if key in ColorTransform._MATRIX_CACHE:
            return ColorTransform._MATRIX_CACHE[key]

        kr, kb = ColorTransform.get_coeffs(colorspace)
        kg = 1.0 - kr - kb

        if direction == "yuv2rgb":
            assert color_range is not None
            # YUV to RGB matrix (for normalized Y, U, V in [0, 1] and centered U, V)
            m = [
                [1.0, 0.0, 2.0 * (1.0 - kr)],
                [1.0, -kb * 2.0 * (1.0 - kb) / kg, -kr * 2.0 * (1.0 - kr) / kg],
                [1.0, 2.0 * (1.0 - kb), 0.0],
            ]
            matrix = torch.tensor(m, device=device, dtype=dtype)

            # Range and normalization adjustment
            offsets: torch.Tensor
            gains: torch.Tensor
            if color_range == ColorRange.JPEG:
                offsets = torch.tensor([0.0, 128.0 / 255.0, 128.0 / 255.0], device=device, dtype=dtype)
                gains = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
            else:
                offsets = torch.tensor(
                    [16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
                    device=device,
                    dtype=dtype,
                )
                gains = torch.tensor(
                    [255.0 / 219.0, 255.0 / 224.0, 255.0 / 224.0],
                    device=device,
                    dtype=dtype,
                )

            gains = gains / div
            weight = matrix * gains.unsqueeze(0)
            bias = -torch.mv(weight * div, offsets)
            weight = weight.view(3, 3, 1, 1)
            res_y2r = (weight, bias)
            ColorTransform._MATRIX_CACHE[key] = res_y2r
            return res_y2r
        else:
            # rgb2yuv
            m = [
                [kr, kg, kb],
                [-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5],
                [0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))],
            ]
            weight_r2y = torch.tensor(m, device=device, dtype=dtype).view(3, 3, 1, 1)
            ColorTransform._MATRIX_CACHE[key] = weight_r2y
            return weight_r2y

    @staticmethod
    def yuv_to_rgb(
        y: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        colorspace: Colorspace | int = Colorspace.ITU709,
        color_range: ColorRange | int = ColorRange.MPEG,
        div: float = 255.0,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        res = ColorTransform._get_matrix(colorspace, y.device, y.dtype, "yuv2rgb", color_range, div)
        assert isinstance(res, tuple)
        weight, bias = res

        # Optimization: Process planes sequentially to minimize peak VRAM.
        # 1. Start with Y plane
        rgb = F.conv2d(y, weight[:, 0:1])

        # 2. Interpolate and add U plane
        if u.shape[-2:] != y.shape[-2:]:
            u = F.interpolate(
                u,
                size=y.shape[-2:],
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            )
        rgb.add_(F.conv2d(u, weight[:, 1:2]))
        del u

        # 3. Interpolate and add V plane
        if v.shape[-2:] != y.shape[-2:]:
            v = F.interpolate(
                v,
                size=y.shape[-2:],
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            )
        rgb.add_(F.conv2d(v, weight[:, 2:3]))
        del v

        # 4. Add Bias and Clamp
        rgb.add_(bias.view(1, 3, 1, 1))
        return rgb.clamp_(0.0, 1.0)

    @staticmethod
    @torch.no_grad()
    def rgb_to_yuv(
        rgb: torch.Tensor,
        colorspace: Colorspace | int = Colorspace.ITU709,
        color_range: ColorRange | int = ColorRange.MPEG,
        out_format: str | None = "yuv444p",
        mode: str = "bilinear",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = ColorTransform._get_matrix(colorspace, rgb.device, rgb.dtype, "rgb2yuv")
        assert isinstance(weight, torch.Tensor)
        yuv_n = F.conv2d(rgb, weight)

        y_n, u_n, v_n = yuv_n[:, 0:1], yuv_n[:, 1:2], yuv_n[:, 2:3]

        y: torch.Tensor
        u: torch.Tensor
        v: torch.Tensor
        if color_range == ColorRange.JPEG:
            y, u, v = y_n, u_n + 0.5, v_n + 0.5
        else:
            y = (y_n * 219.0 + 16.0) / 255.0
            u = (u_n * 224.0 + 128.0) / 255.0
            v = (v_n * 224.0 + 128.0) / 255.0

        if out_format:
            h, w = y.shape[-2:]
            if out_format in {"yuv420p", "yuvj420p", "yuv420p10le", "nv12", "p010le"}:
                if mode == "nearest":
                    u = F.interpolate(u, size=(h // 2, w // 2), mode="nearest")
                    v = F.interpolate(v, size=(h // 2, w // 2), mode="nearest")
                else:
                    u = F.interpolate(u, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
                    v = F.interpolate(v, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
            elif out_format in {"yuv422p", "yuvj422p", "yuv422p10le"}:
                if mode == "nearest":
                    u = F.interpolate(u, size=(h, w // 2), mode="nearest")
                    v = F.interpolate(v, size=(h, w // 2), mode="nearest")
                else:
                    u = F.interpolate(u, size=(h, w // 2), mode="bilinear", align_corners=False)
                    v = F.interpolate(v, size=(h, w // 2), mode="bilinear", align_corners=False)

        return y, u, v

    @staticmethod
    def to_yuv_planes(
        y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, out_format: str = "nv12"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert float Y, U, V tensors to output format planes (tensors)."""
        if out_format == "nv12":
            # Y: [0, 1] -> uint8 [0, 255]
            y_out = (y * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
            # UV: Interleave U and V
            # u, v are (B, 1, H/2, w/2)
            uv_out = torch.stack([u, v], dim=-1)  # (B, 1, H/2, W/2, 2)
            uv_out = (uv_out * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
            return y_out.squeeze(0).squeeze(0), uv_out.squeeze(0).squeeze(0)
        elif out_format == "p010le":
            # Y: [0, 1] -> uint16 [0, 1023 << 6]
            # 1023 << 6 = 65472
            y_out = (y * 65472.0 + 0.5).clamp(0, 65472).to(torch.uint16)
            # UV: Interleave U and V
            uv_out = torch.stack([u, v], dim=-1)  # (B, 1, H/2, W/2, 2)
            uv_out = (uv_out * 65472.0 + 0.5).clamp(0, 65472).to(torch.uint16)
            return y_out.squeeze(0).squeeze(0), uv_out.squeeze(0).squeeze(0)
        raise ValueError(f"Unsupported output format for to_yuv_planes: {out_format}")


class TensorFrame:
    planes: torch.Tensor
    pts: int | None
    dts: int | None
    time_base: Fraction | None
    colorspace: Colorspace | int
    color_primaries: ColorPrimaries | int
    color_trc: ColorTrc | int
    color_range: ColorRange | int
    side_data: Any
    use_16bit: bool

    def __init__(
        self,
        planes: torch.Tensor,
        pts: int | None,
        dts: int | None,
        time_base: Fraction | None,
        colorspace: Colorspace | int,
        color_primaries: ColorPrimaries | int,
        color_trc: ColorTrc | int,
        color_range: ColorRange | int,
        side_data: Any,
        use_16bit: bool,
    ) -> None:
        self.planes = planes
        self.pts = pts
        self.dts = dts
        self.time_base = time_base
        self.colorspace = colorspace
        self.color_primaries = color_primaries
        self.color_trc = color_trc
        self.color_range = color_range
        self.side_data = side_data
        self.use_16bit = use_16bit

    @property
    def width(self) -> int:
        return self.planes.shape[-1]

    @property
    def height(self) -> int:
        return self.planes.shape[-2]

    def to(self, *args: Any, **kwargs: Any) -> "TensorFrame":
        self.planes = self.planes.to(*args, **kwargs)
        return self

    def to_bchw(self) -> torch.Tensor:
        if self.planes.ndim == 4:
            return self.planes
        else:
            assert self.planes.ndim == 3
            return self.planes.unsqueeze(0)

    def to_chw(self) -> torch.Tensor:
        if self.planes.ndim == 3:
            return self.planes
        else:
            assert self.planes.ndim == 4 and self.planes.shape[0] == 1
            return self.planes.squeeze(0)


def pix_fmt_requires_16bit(pix_fmt: str) -> bool:
    return pix_fmt in {
        "yuv420p10le",
        "p010le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
        "yuv444p16le",
        "gbrp16le",
        "gbrp12le",
        "gbrp10le",
        "rgb48le",
    }


class InputTransform:
    # Convert av.VideoFrame to RGB Full Range tensor/ndarray
    # For src_*, use VideoMetadata class

    HW_DEVICE_TYPES: Set[str] = set([t.name for t in HWDeviceType if t.name != "none"])
    src_pix_fmt: str
    src_colorspace: Colorspace | int
    src_color_primaries: ColorPrimaries | int
    src_color_trc: ColorTrc | int
    src_color_range: ColorRange | int
    dst_colorspace: Colorspace | int
    dst_color_primaries: ColorPrimaries | int
    dst_color_trc: ColorTrc | int
    dst_color_range: ColorRange | int
    use_16bit: bool
    device: torch.device
    dtype: torch.dtype
    rgb_format: str

    def __init__(
        self,
        src_pix_fmt: str,
        src_colorspace: Colorspace | int,
        src_color_primaries: ColorPrimaries | int,
        src_color_trc: ColorTrc | int,
        src_color_range: ColorRange | int,
        dst_colorspace: Colorspace | int,
        dst_color_primaries: ColorPrimaries | int,
        dst_color_trc: ColorTrc | int,
        dst_color_range: ColorRange | int,
        use_16bit: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.src_pix_fmt = src_pix_fmt
        self.src_colorspace = src_colorspace
        self.src_color_primaries = src_color_primaries
        self.src_color_trc = src_color_trc
        self.src_color_range = src_color_range
        self.dst_colorspace = dst_colorspace
        self.dst_color_primaries = dst_color_primaries
        self.dst_color_trc = dst_color_trc
        self.dst_color_range = dst_color_range
        self.use_16bit = use_16bit
        self.device = device
        self.dtype = dtype
        self.rgb_format = get_rgb_pix_fmt(use_16bit)

    def to_tensor_hw(self, frame: av.VideoFrame) -> TensorFrame:
        rgb = ColorTransform.to_tensor(
            frame,
            dst_colorspace=self.dst_colorspace,
            dst_color_range=ColorRange.JPEG,
            device=self.device,
            dtype=self.dtype,
        )
        return TensorFrame(
            planes=rgb,
            pts=frame.pts,
            dts=frame.dts,
            time_base=frame.time_base,
            colorspace=self.dst_colorspace,
            color_primaries=self.dst_color_primaries,
            color_trc=self.dst_color_trc,
            color_range=ColorRange.JPEG,
            side_data=frame.side_data,
            use_16bit=self.use_16bit,
        )

    def to_tensor_av(self, frame: av.VideoFrame) -> TensorFrame:
        frame = frame.reformat(
            format=self.rgb_format,
            src_colorspace=self.src_colorspace,
            src_color_range=self.src_color_range,
            dst_colorspace=self.dst_colorspace,
            dst_color_primaries=self.dst_color_primaries,
            dst_color_trc=self.dst_color_trc,
            dst_color_range=ColorRange.JPEG,  # full range
        )
        # Type ignored because to_ndarray is Any
        rgb_np: np.ndarray = frame.to_ndarray(format=self.rgb_format)
        rgb = torch.from_numpy(rgb_np).contiguous()
        rgb = rgb.to(self.device)
        # Normalize based on bit depth
        if rgb_np.dtype == np.uint8:
            rgb = rgb.permute(2, 0, 1).to(self.dtype) / 255.0
        elif rgb_np.dtype == np.uint16:
            rgb = rgb.permute(2, 0, 1).to(self.dtype) / 65535.0
        else:
            raise ValueError(f"Unsupported dtype from to_ndarray: {rgb_np.dtype}")

        rgb = rgb.contiguous()

        return TensorFrame(
            planes=rgb,
            pts=frame.pts,
            dts=frame.dts,
            time_base=frame.time_base,
            colorspace=self.dst_colorspace,
            color_primaries=self.dst_color_primaries,
            color_trc=self.dst_color_trc,
            color_range=ColorRange.JPEG,
            side_data=frame.side_data,
            use_16bit=self.use_16bit,
        )

    def transform(self, frame: av.VideoFrame) -> TensorFrame:
        # Check source metadata
        assert frame.colorspace == self.src_colorspace or frame.colorspace == COLORSPACE_UNSPECIFIED
        assert frame.color_range == self.src_color_range or frame.color_range == ColorRange.UNSPECIFIED

        if frame.format.name == "cuda" and self.src_pix_fmt in {
            "yuv420p",
            "yuv420p10le",
        }:
            # expected cuda with nv12 or p010le
            return self.to_tensor_hw(frame)

        if frame.format.name in InputTransform.HW_DEVICE_TYPES:
            # hw download
            frame = frame.reformat()
        assert len(frame.format.components) > 0, f"Unexpected (hw) format {frame.format}"

        return self.to_tensor_av(frame)

    def __call__(self, frame: av.VideoFrame) -> TensorFrame:
        return self.transform(frame)


class OutputTransform:
    dst_pix_fmt: str
    dst_colorspace: Colorspace | int
    dst_color_primaries: ColorPrimaries | int
    dst_color_trc: ColorTrc | int
    dst_color_range: ColorRange | int
    use_16bit: bool
    cuda_context: CudaContext | None

    # Convert tensor/ndarray to av.VideoFrame
    def __init__(
        self,
        dst_pix_fmt: str,
        dst_colorspace: Colorspace | int,
        dst_color_primaries: ColorPrimaries | int,
        dst_color_trc: ColorTrc | int,
        dst_color_range: ColorRange | int,
        cuda_context: CudaContext | None = None,
    ) -> None:
        self.dst_pix_fmt = dst_pix_fmt
        self.dst_colorspace = dst_colorspace
        self.dst_color_primaries = dst_color_primaries
        self.dst_color_trc = dst_color_trc
        self.dst_color_range = dst_color_range
        self.use_16bit = pix_fmt_requires_16bit(dst_pix_fmt)
        self.cuda_context = cuda_context

    def from_video_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame = frame.reformat(
            format=self.dst_pix_fmt,
            src_colorspace=self.dst_colorspace,
            src_color_range=ColorRange.JPEG,
            dst_colorspace=self.dst_colorspace,
            dst_color_primaries=self.dst_color_primaries,
            dst_color_trc=self.dst_color_trc,
            dst_color_range=self.dst_color_range,
        )
        return frame

    def from_ndarray(self, x: np.ndarray) -> av.VideoFrame:
        format: str
        if x.dtype == np.uint8:
            format = "rgb24"
        elif x.dtype == np.uint16:
            format = "gbrp16le"
        else:
            raise ValueError(f"unsupported dtype {x.dtype}")

        frame = av.VideoFrame.from_ndarray(x, format=format)
        return self.from_video_frame(frame)

    def from_tensor(self, x: torch.Tensor) -> av.VideoFrame:
        dtype: Any
        value_scale: float
        if self.use_16bit:
            dtype = torch.uint16
            value_scale = 65535.0
        else:
            dtype = torch.uint8
            value_scale = 255.0
        x_nd = (x.permute(1, 2, 0).contiguous() * value_scale).round().to(dtype).detach().cpu().numpy()
        return self.from_ndarray(x_nd)

    def is_dlpack_supported(self, x: torch.Tensor) -> bool:
        return (
            self.cuda_context is not None
            and is_nvidia_gpu(x.device)
            and self.dst_pix_fmt in {"nv12", "p010le", "yuv420p", "yuv420p10le"}
        )

    def from_cuda_tensor(self, x: torch.Tensor) -> av.VideoFrame:
        assert self.is_dlpack_supported(x)

        internal_pix_fmt: str
        if self.dst_pix_fmt == "yuv420p":
            internal_pix_fmt = "nv12"
        elif self.dst_pix_fmt == "yuv420p10le":
            internal_pix_fmt = "p010le"
        else:
            internal_pix_fmt = self.dst_pix_fmt

        y, u, v = ColorTransform.rgb_to_yuv(
            x if x.dim() == 4 else x.unsqueeze(0),
            colorspace=self.dst_colorspace,
            color_range=self.dst_color_range,
            out_format=internal_pix_fmt,
        )
        y_p, uv_p = ColorTransform.to_yuv_planes(y, u, v, out_format=internal_pix_fmt)

        # Ensure PyTorch has finished writing to y_p and uv_p
        if torch.cuda.default_stream() != torch.cuda.current_stream():
            torch.cuda.current_stream().synchronize()

        frame = av.VideoFrame.from_dlpack(
            (y_p, uv_p),
            format=internal_pix_fmt,
            primary_ctx=True,
            cuda_context=self.cuda_context,
        )
        frame.pts = None
        frame.dts = None
        frame.colorspace = self.dst_colorspace
        frame.color_primaries = self.dst_color_primaries
        frame.color_trc = self.dst_color_trc
        frame.color_range = self.dst_color_range
        return frame

    def transform(self, x: av.VideoFrame | torch.Tensor | np.ndarray | TensorFrame) -> av.VideoFrame:
        if isinstance(x, TensorFrame):
            # For simplicity, extract planes. PTS/DTS handling can be added later if needed.
            x = x.planes

        if isinstance(x, av.VideoFrame):
            return self.from_video_frame(x)
        elif isinstance(x, np.ndarray):
            return self.from_ndarray(x)

        if not (torch.is_tensor(x) and x.dtype in (torch.float32, torch.float16)):
            raise ValueError(f"Unsupported frame type: {type(x)}")

        if self.is_dlpack_supported(x):
            return self.from_cuda_tensor(x)

        return self.from_tensor(x)

    def __call__(self, x: av.VideoFrame | torch.Tensor | np.ndarray | TensorFrame) -> av.VideoFrame:
        return self.transform(x)


def setup_color_transform(
    sw_format: VideoMetadata | None,
    config: Any,
) -> Tuple[Dict[str, Any], OutputTransform]:
    """Calculate color configuration and transform objects based on user config."""
    pix_fmt: str
    colorspace: Colorspace | int
    color_primaries: ColorPrimaries | int
    color_trc: ColorTrc | int
    color_range: ColorRange | int
    input_reformat_options: Dict[str, Any]

    if sw_format:
        (
            pix_fmt,
            colorspace,
            color_primaries,
            color_trc,
            color_range,
        ) = sw_format.get_target_colorspace(config.colorspace, config.pix_fmt)
        input_reformat_options = sw_format.get_input_reformat_options(colorspace, color_primaries, color_trc)
    else:
        exported_output_colorspace: int = (
            config.output_colorspace if config.output_colorspace is not None else COLORSPACE_UNSPECIFIED
        )
        exported_output_color_primaries: int = (
            config.output_color_primaries
            if config.output_color_primaries is not None
            else int(ColorPrimaries.UNSPECIFIED)
        )
        exported_output_color_trc: int = (
            config.output_color_trc if config.output_color_trc is not None else int(ColorTrc.UNSPECIFIED)
        )
        exported_source_color_range: int = (
            config.source_color_range if config.source_color_range is not None else int(ColorRange.UNSPECIFIED)
        )
        (
            pix_fmt,
            colorspace,
            color_primaries,
            color_trc,
            color_range,
        ) = VideoMetadata.get_target_colorspace_static(
            colorspace_mode=config.colorspace,
            pix_fmt=config.pix_fmt,
            src_colorspace=exported_output_colorspace,
            src_color_primaries=exported_output_color_primaries,
            src_color_trc=exported_output_color_trc,
            src_color_range=exported_source_color_range,
        )
        input_reformat_options = {
            "src_colorspace": colorspace,
            "src_color_primaries": color_primaries,
            "src_color_trc": color_trc,
            "src_color_range": color_range,
            "dst_colorspace": colorspace,
            "dst_color_primaries": color_primaries,
            "dst_color_trc": color_trc,
            "dst_color_range": ColorRange.JPEG,
        }

    cuda_context = None
    if (
        is_nvidia_gpu(config.device)
        and config.video_codec in {"h264_nvenc", "hevc_nvenc"}
        and config.pix_fmt in {"nv12", "p010le"}
    ):
        device_id = config.device.index if config.device is not None else None
        device_id = device_id if device_id is not None else 0
        cuda_context = CudaContext(device_id=device_id, primary_ctx=True)

    output_reformatter = OutputTransform(
        dst_pix_fmt=pix_fmt,
        dst_colorspace=colorspace,
        dst_color_primaries=color_primaries,
        dst_color_trc=color_trc,
        dst_color_range=color_range,
        cuda_context=cuda_context,
    )

    return input_reformat_options, output_reformatter


def configure_video_codec(config: Any) -> None:
    """Adjust pixel format and codec based on hardware/software encoder constraints."""
    codec: str = config.video_codec
    pix_fmt: str = config.pix_fmt

    # UtVideo constraints
    if codec == "utvideo":
        if pix_fmt == "rgb24":
            config.pix_fmt = "gbrp"
        # Only TV range is reliably supported by UtVideo
        if config.colorspace in {"bt601", "bt601-pc", "bt601-tv"}:
            config.colorspace = "bt601-tv"
        elif config.colorspace in {"bt709", "bt709-pc", "bt709-tv"}:
            config.colorspace = "bt709-tv"
        elif config.colorspace in {
            "bt2020",
            "bt2020-pc",
            "bt2020-tv",
            "bt2020-pq-tv",
            "auto",
            "copy",
        }:
            config.colorspace = "bt709-tv"

    # FFV1 constraints
    elif codec == "ffv1":
        if pix_fmt == "rgb24":
            config.pix_fmt = "bgr0"

    # libx264/libx265 constraints
    elif codec == "libx264":
        if pix_fmt in {"rgb24", "gbrp"}:
            config.video_codec = "libx264rgb"
            config.pix_fmt = "rgb24"
    elif codec in {"libx265", "h264_nvenc", "hevc_nvenc"}:
        if pix_fmt == "rgb24":
            config.pix_fmt = "gbrp"

    # Hardware acceleration specific mappings (NVENC, QSV)
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        if pix_fmt == "yuv420p":
            config.pix_fmt = "nv12"
        elif pix_fmt == "yuv420p10le":
            config.pix_fmt = "p010le"

    if codec in {"h264_qsv", "hevc_qsv"}:
        if pix_fmt == "yuv420p":
            config.pix_fmt = "nv12"
        elif pix_fmt == "yuv420p10le":
            config.pix_fmt = "p010le"


def _test_configure() -> None:
    class MockConfig:
        def __init__(
            self,
            colorspace: str = "auto",
            pix_fmt: str = "yuv420p",
            video_codec: str = "libx264",
        ) -> None:
            self.colorspace = colorspace
            self.pix_fmt = pix_fmt
            self.video_codec = video_codec
            self.output_colorspace = 2
            self.output_color_primaries = 2
            self.output_color_trc = 2
            self.source_color_range = 2
            self.state_updated = lambda c: print("Config updated callback triggered")
            self.device = None

    class MockSWFormat(VideoMetadata):
        def __init__(
            self,
            colorspace: int = 2,
            color_range: ColorRange = ColorRange.UNSPECIFIED,
            height: int = 1080,
        ) -> None:
            self.colorspace: Any = colorspace
            self.color_range: Any = color_range
            self.height: int = height
            self.format: Any = type("Format", (), {"name": "yuv420p"})
            self.color_primaries: Any = 2
            self.color_trc: Any = 2

    print("--- Start configure tests ---")
    sw_hd: Any = MockSWFormat(height=1080)

    print("Testing setup_color_transform (Auto HD)...")
    cfg = MockConfig(colorspace="auto", pix_fmt="yuv420p")
    input_reformat_options, output_reformatter = setup_color_transform(sw_hd, cfg)
    cfg.state_updated(cfg)
    assert int(output_reformatter.dst_colorspace) == 1
    assert output_reformatter.dst_pix_fmt == "yuv420p"
    assert input_reformat_options["dst_color_primaries"] == 1

    print("Testing setup_color_transform (bt709-pc)...")
    cfg = MockConfig(colorspace="bt709-pc", pix_fmt="yuv420p")
    input_reformat_options, output_reformatter = setup_color_transform(sw_hd, cfg)
    cfg.state_updated(cfg)
    assert output_reformatter.dst_pix_fmt == "yuvj420p"
    assert input_reformat_options["src_color_range"] == 1
    assert input_reformat_options["dst_color_trc"] == 1

    print("Testing setup_color_transform (explicit unspecified)...")
    cfg = MockConfig(colorspace="unspecified", pix_fmt="yuv420p")
    input_reformat_options, output_reformatter = setup_color_transform(sw_hd, cfg)
    cfg.state_updated(cfg)
    assert int(output_reformatter.dst_colorspace) == 2
    assert output_reformatter.dst_pix_fmt == "yuv420p"

    print("Testing setup_color_transform (RGB output)...")
    cfg = MockConfig(colorspace="auto", pix_fmt="rgb24")
    input_reformat_options, output_reformatter = setup_color_transform(sw_hd, cfg)
    cfg.state_updated(cfg)
    assert int(output_reformatter.dst_colorspace) == 1

    print("Testing configure_video_codec...")
    cfg_nv = MockConfig(pix_fmt="yuv420p10le", video_codec="h264_nvenc")
    configure_video_codec(cfg_nv)
    assert cfg_nv.pix_fmt == "p010le"
    print("OK")
    print("--- End configure tests ---")


if __name__ == "__main__":
    _test_configure()
