import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Any

import wx
import wx.lib.stattext as stattext
from wx.lib.buttons import GenBitmapButton
from wx.lib.intctrl import IntCtrl

from nunif.gui import EditableComboBox, load_icon, refresh_layouts, validate_number

BACKENDS_WITH_TARGET = {"wc_mp", "wc_cuda", "mss", "kwcapture"}
BACKENDS_WITH_CROP = BACKENDS_WITH_TARGET | {"obs"}
ADD_DEBUG_METHOD = bool(os.getenv("DEBUG", False))


def is_draw_cursor_supported(backend: str) -> bool:
    if backend in {"pil", "kwcapture", "obs"}:
        return True
    if backend == "mss" and sys.platform != "linux":
        return True
    return False


myEVT_FPS = wx.NewEventType()
EVT_FPS = wx.PyEventBinder(myEVT_FPS, 0)


class FPSEvent(wx.PyCommandEvent):
    def __init__(
        self,
        etype: int,
        eid: int,
        estimated_fps: float | None = None,
        screenshot_fps: float | None = None,
        streaming_fps: float | None = None,
        screen_size: Any = None,
        url: str | None = None,
    ) -> None:
        super().__init__(etype, eid)
        self.estimated_fps = estimated_fps
        self.screenshot_fps = screenshot_fps
        self.streaming_fps = streaming_fps
        self.screen_size = screen_size
        self.url = url

    def GetValue(self) -> tuple[float | None, float | None, float | None, Any, str | None]:
        return (self.estimated_fps, self.screenshot_fps, self.streaming_fps, self.screen_size, self.url)


class FPSGUI:
    def __init__(self, parent: wx.Window, **kwargs: Any) -> None:
        self.parent = parent

    def set_url(self, url: str | None) -> None:
        wx.PostEvent(self.parent, FPSEvent(myEVT_FPS, -1, None, None, None, None, url))

    def update(self, estimated_fps: float, screenshot_fps: float, streaming_fps: float, screen_size: Any) -> None:
        wx.PostEvent(
            self.parent,
            FPSEvent(myEVT_FPS, -1, estimated_fps, screenshot_fps, streaming_fps, screen_size, None),
        )


class DesktopStreamGUIMixin:
    """Shared wxPython controls, event handlers, and arg parsing for desktop streaming."""

    depth_model: Any
    depth_model_type: str | None
    depth_model_device_id: list[int] | None
    depth_model_height: int | None
    txt_port: IntCtrl

    if TYPE_CHECKING:

        def FromDIP(self, sz: tuple[int, int]) -> wx.Size: ...
        def SetStatusText(self, text: str, number: int = 0) -> None: ...
        def GetSizer(self) -> wx.Sizer: ...
        def IsShown(self) -> bool: ...

    def _T(self, s: str) -> str:
        return s

    def _init_desktop_deps(self) -> None:
        if getattr(self, "_desktop_deps_loaded", False):
            return

        import torch

        from iw3.base_depth_model import BaseDepthModel
        from iw3.desktop.utils import (
            ENABLE_GPU_JPEG,
            WAYLAND_CAPTURE_MISSING_MESSAGE,
            enum_window_names,
            get_monitor_size_list,
            is_kwcapture_supported,
            is_mss_supported,
            is_obs_supported,
            is_wayland,
            read_runtime_adjustment_values,
        )
        from nunif.device import create_device, mps_is_available, xpu_is_available
        from nunif.initializer import gc_collect
        from nunif.models.utils import check_compile_support

        self._torch = torch
        self._BaseDepthModel = BaseDepthModel
        self._enum_window_names = enum_window_names
        self._get_monitor_size_list = get_monitor_size_list
        self._is_mss_supported = is_mss_supported
        self._is_wayland = is_wayland
        self._wayland_missing_msg = WAYLAND_CAPTURE_MISSING_MESSAGE
        self._create_device = create_device
        self._mps_is_available = mps_is_available
        self._xpu_is_available = xpu_is_available
        self._check_compile_support = check_compile_support
        self._gc_collect = gc_collect
        self._enable_gpu_jpeg = ENABLE_GPU_JPEG
        self._read_runtime_adjustment_values = read_runtime_adjustment_values
        self._has_obs = is_obs_supported()
        self._has_kwcapture = is_kwcapture_supported()
        self._desktop_deps_loaded = True

    def _build_stream_network_controls(
        self,
        parent: wx.Window,
        grid: wx.GridBagSizer,
        start_row: int,
    ) -> list[wx.Window]:
        self._init_desktop_deps()
        T = self._T

        self.sep_network2 = wx.StaticLine(parent, style=wx.LI_HORIZONTAL)
        self.lbl_stream_fps = wx.StaticText(parent, label=T("Streaming FPS"))
        self.cbo_stream_fps = EditableComboBox(parent, choices=["60", "30", "24", "15", "8"], name="cbo_stream_fps")
        self.cbo_stream_fps.SetSelection(1)

        self.lbl_stream_height = wx.StaticText(parent, label=T("Streaming Resolution"))
        self.cbo_stream_height = EditableComboBox(parent, choices=["1080", "720"], name="cbo_stream_height")
        self.cbo_stream_height.SetSelection(0)

        self.lbl_stream_quality = wx.StaticText(parent, label=T("MJPEG Quality"))
        self.cbo_stream_quality = EditableComboBox(
            parent, choices=["100", "95", "90", "85", "80"], name="cbo_stream_quality"
        )
        self.cbo_stream_quality.SetSelection(2)

        self.chk_gpu_jpeg = wx.CheckBox(parent, label=T("GPU JPEG"), name="chk_gpu_jpeg")
        self.chk_gpu_jpeg.SetValue(False)
        if not self._enable_gpu_jpeg:
            self.chk_gpu_jpeg.Disable()

        r = start_row
        grid.Add(self.sep_network2, (r, 0), (1, 3), flag=wx.EXPAND | wx.ALL)
        grid.Add(self.lbl_stream_fps, (r + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.cbo_stream_fps, (r + 1, 1), flag=wx.EXPAND)
        grid.Add(self.lbl_stream_height, (r + 2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.cbo_stream_height, (r + 2, 1), flag=wx.EXPAND)
        grid.Add(self.lbl_stream_quality, (r + 3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.cbo_stream_quality, (r + 3, 1), flag=wx.EXPAND)
        grid.Add(self.chk_gpu_jpeg, (r + 4, 1), flag=wx.ALIGN_CENTER_VERTICAL)

        return [
            self.sep_network2,
            self.lbl_stream_fps,
            self.cbo_stream_fps,
            self.lbl_stream_height,
            self.cbo_stream_height,
            self.lbl_stream_quality,
            self.cbo_stream_quality,
            self.chk_gpu_jpeg,
        ]

    def _build_stereo_group(
        self,
        parent: wx.Window,
        warning_font: wx.Font,
        warning_color: tuple[int, int, int],
        extra_layout_fn: Any = None,
    ) -> wx.StaticBoxSizer:
        self._init_desktop_deps()
        T = self._T

        self.grp_stereo = wx.StaticBox(parent, label=T("Stereo Generation"))
        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = EditableComboBox(
            self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"], name="cbo_divergence"
        )
        self.lbl_divergence_warning = stattext.GenStaticText(self.grp_stereo, label="")
        self.lbl_divergence_warning.SetFont(warning_font)
        self.lbl_divergence_warning.SetForegroundColour(warning_color)
        self.lbl_divergence_warning.Hide()

        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(5)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence_mode = wx.ComboBox(
            self.grp_stereo, choices=["constant", "sod_v1"], name="cbo_convergence_mode"
        )
        self.cbo_convergence_mode.SetEditable(False)
        self.cbo_convergence_mode.SetSelection(0)

        self.cbo_convergence = EditableComboBox(
            self.grp_stereo, choices=["0.0", "0.25", "0.5", "1.0"], name="cbo_convergence"
        )
        self.cbo_convergence.SetSelection(2)
        self.cbo_convergence.SetToolTip("Convergence")

        self.lbl_synthetic_view = wx.StaticText(self.grp_stereo, label=T("Synthetic View"))
        self.cbo_synthetic_view = wx.ComboBox(
            self.grp_stereo, choices=["both", "right", "left"], name="cbo_synthetic_view"
        )
        self.cbo_synthetic_view.SetEditable(False)
        self.cbo_synthetic_view.SetSelection(0)

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(
            self.grp_stereo,
            choices=[
                "mlbw_l2",
                "mlbw_l4",
                "mlbw_l2s",
                "row_flow_v3",
                "row_flow_v3_sym",
                "row_flow_v2",
                "forward_fill",
                "monobw",
            ]
            + (["NULL"] if ADD_DEBUG_METHOD else []),
            name="cbo_method",
        )
        self.cbo_method.SetEditable(False)
        self.cbo_method.SetSelection(4)

        self.chk_small_model_only = wx.CheckBox(
            self.grp_stereo, label=T("List small model only"), name="chk_small_model_only"
        )
        self.chk_small_model_only.SetValue(True)
        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        depth_models = self.get_depth_models(small_only=False)
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo, choices=depth_models, name="cbo_depth_model")
        self.cbo_depth_model.SetEditable(False)
        self.cbo_depth_model.SetSelection(depth_models.index("Any_V2_S"))

        self.lbl_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_resolution = EditableComboBox(self.grp_stereo, choices=["Default", "512"], name="cbo_zoed_resolution")
        self.cbo_resolution.SetSelection(0)

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(
            self.grp_stereo, choices=["-3", "-2", "-1", "0", "1", "2", "3"], name="cbo_foreground_scale"
        )
        self.cbo_foreground_scale.SetSelection(3)

        self.lbl_edge_dilation = wx.StaticText(self.grp_stereo, label=T("Edge Fix"), name="lbl_edge_dilation")
        self.cbo_edge_dilation = EditableComboBox(
            self.grp_stereo, choices=["0", "1", "2", "3", "4"], size=self.FromDIP((90, -1)), name="cbo_edge_dilation"
        )
        self.cbo_edge_dilation_y = EditableComboBox(
            self.grp_stereo, choices=["", "0", "1", "2"], name="cbo_edge_dilation_y"
        )
        self.cbo_edge_dilation.SetSelection(2)
        self.cbo_edge_dilation_y.SetSelection(0)
        self.lbl_edge_dilation.SetToolTip(T("Reduce distortion of foreground and background edges"))
        self.cbo_edge_dilation.SetToolTip(T("X or XY"))
        self.cbo_edge_dilation_y.SetToolTip(T("Y"))

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo, label=T("Flicker Reduction"), name="chk_ema_normalize")
        self.chk_ema_normalize.SetValue(True)
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))
        self.cbo_ema_decay = EditableComboBox(
            self.grp_stereo, choices=["0.99", "0.95", "0.9", "0.75", "0.5"], name="cbo_ema_decay"
        )
        self.cbo_ema_decay.SetSelection(2)

        self.chk_preserve_screen_border = wx.CheckBox(
            self.grp_stereo, label=T("Preserve Screen Border"), name="chk_preserve_screen_border"
        )
        self.chk_preserve_screen_border.SetValue(True)
        self.chk_preserve_screen_border.SetToolTip(T("Force set screen border parallax to zero"))

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(
            self.grp_stereo,
            choices=[
                "Half SBS",
                "Full SBS",
                "Half TB",
                "Full TB",
                "Cross Eyed",
                "RGB-D",
                "Half RGB-D",
                "Anaglyph",
            ],
            name="cbo_stereo_format",
        )
        self.cbo_stereo_format.SetEditable(False)
        self.cbo_stereo_format.SetSelection(0)

        self.lbl_anaglyph_method = wx.StaticText(self.grp_stereo, label=T("Anaglyph Method"))
        self.cbo_anaglyph_method = wx.ComboBox(
            self.grp_stereo,
            choices=["dubois", "dubois2", "color", "gray", "half-color", "wimmer", "wimmer2"],
            name="cbo_anaglyph_method",
        )
        self.cbo_anaglyph_method.SetEditable(False)
        self.cbo_anaglyph_method.SetSelection(0)
        self.lbl_anaglyph_method.Hide()
        self.cbo_anaglyph_method.Hide()

        self.lbl_format_device = wx.StaticText(self.grp_stereo, label=T(""))
        self.chk_pad_16_9 = wx.CheckBox(self.grp_stereo, label=T("Padding") + " 16:9", name="chk_pad_16_9")
        self.chk_pad_16_9.SetValue(False)

        self.lbl_autocrop = wx.StaticText(self.grp_stereo, label=T("AutoCrop"))
        self.cbo_autocrop = wx.ComboBox(
            self.grp_stereo, choices=["", "BLACK_TB", "BLACK", "FLAT_TB", "FLAT"], name="cbo_autocrop"
        )
        self.cbo_autocrop.SetEditable(False)
        self.cbo_autocrop.SetSelection(0)

        stereo_layout = wx.GridBagSizer(vgap=4, hgap=4)
        stereo_layout.SetEmptyCellSize((0, 0))
        i = 0
        stereo_layout.Add(self.lbl_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_divergence, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(
            self.lbl_divergence_warning,
            pos=(i := i + 1, 0),
            span=(0, 3),
            flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL,
        )
        stereo_layout.Add(self.lbl_convergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_convergence_mode, (i, 1), flag=wx.EXPAND)
        stereo_layout.Add(self.cbo_convergence, (i, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_synthetic_view, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_synthetic_view, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_method, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.chk_small_model_only, (i := i + 1, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.lbl_depth_model, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_depth_model, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_resolution, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_resolution, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_foreground_scale, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_foreground_scale, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_edge_dilation, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_edge_dilation, (i, 1), flag=wx.EXPAND)
        stereo_layout.Add(self.cbo_edge_dilation_y, (i, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.chk_ema_normalize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_ema_decay, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.chk_preserve_screen_border, (i := i + 1, 0), (1, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.lbl_stereo_format, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_stereo_format, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_anaglyph_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_anaglyph_method, (i, 1), (1, 2), flag=wx.EXPAND)
        stereo_layout.Add(self.lbl_format_device, (i := i + 1, 1), (1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.chk_pad_16_9, (i := i + 1, 0), (1, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.lbl_autocrop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        stereo_layout.Add(self.cbo_autocrop, (i, 1), (1, 2), flag=wx.EXPAND)

        if extra_layout_fn is not None:
            extra_layout_fn(self.grp_stereo, stereo_layout, i)

        self.sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        self.sizer_stereo.Add(stereo_layout, 1, wx.ALL | wx.EXPAND, 4)
        return self.sizer_stereo

    def _build_processor_group(self, parent: wx.Window) -> wx.StaticBoxSizer:
        self._init_desktop_deps()
        T = self._T

        has_windows_capture = bool(importlib.util.find_spec("windows_capture"))
        is_rocm = getattr(self._torch.version, "hip", None) is not None
        has_windows_capture_cuda = (
            bool(importlib.util.find_spec("wc_cuda")) and self._torch.cuda.is_available() and not is_rocm
        )

        self.grp_processor = wx.StaticBox(parent, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=self.FromDIP((200, -1)), name="cbo_device")
        self.cbo_device.SetEditable(False)
        if self._torch.cuda.is_available():
            for dev_idx in range(self._torch.cuda.device_count()):
                device_name = self._torch.cuda.get_device_properties(dev_idx).name
                self.cbo_device.Append(f"{dev_idx}:{device_name}", dev_idx)
        elif self._mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif self._xpu_is_available():
            for dev_idx in range(self._torch.xpu.device_count()):
                device_name = self._torch.xpu.get_device_name(dev_idx)
                self.cbo_device.Append(f"{dev_idx}:{device_name}", dev_idx)

        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_screenshot = wx.StaticText(self.grp_processor, label=T("Screenshot"))
        screenshot_backends = ["pil"]
        if self._is_mss_supported():
            screenshot_backends += ["mss"]
        if has_windows_capture:
            screenshot_backends += ["wc_mp"]
        if has_windows_capture_cuda:
            screenshot_backends += ["wc_cuda"]
        if self._has_kwcapture:
            screenshot_backends += ["kwcapture"]
        if self._has_obs:
            screenshot_backends += ["obs"]
        self.cbo_screenshot = wx.ComboBox(self.grp_processor, choices=screenshot_backends, name="cbo_screenshot")
        self.cbo_screenshot.SetEditable(False)
        default_screenshot_index = 0
        if sys.platform == "win32":
            if "wc_cuda" in screenshot_backends:
                default_screenshot_index = screenshot_backends.index("wc_cuda")
            elif "wc_mp" in screenshot_backends:
                default_screenshot_index = screenshot_backends.index("wc_mp")
        elif self._has_kwcapture and "kwcapture" in screenshot_backends:
            default_screenshot_index = screenshot_backends.index("kwcapture")
        elif self._has_obs and "obs" in screenshot_backends:
            default_screenshot_index = screenshot_backends.index("obs")
        elif sys.platform == "darwin" and "mss" in screenshot_backends:
            default_screenshot_index = screenshot_backends.index("mss")
        self.cbo_screenshot.SetSelection(default_screenshot_index)

        self.lbl_monitor_index = wx.StaticText(self.grp_processor, label=T("Monitor Index"))
        self.cbo_monitor_index = wx.ComboBox(
            self.grp_processor,
            choices=[str(i) for i in range(len(self._get_monitor_size_list()))],
            name="cbo_monitor_index",
        )
        self.cbo_monitor_index.SetEditable(False)
        self.cbo_monitor_index.SetSelection(0)

        self.lbl_window_name = wx.StaticText(self.grp_processor, label=T("Window Name"))
        self.cbo_window_name = EditableComboBox(
            self.grp_processor, size=self.FromDIP((200, -1)), choices=[""], name="cbo_window_name"
        )
        self.cbo_window_name.SetSelection(0)
        self.btn_reload_window_name = GenBitmapButton(self.grp_processor, bitmap=load_icon("view-refresh.png"))
        self.btn_reload_window_name.SetToolTip(T("Reload"))

        self.lbl_crop_region = wx.StaticText(self.grp_processor, label=T("Crop Region (px)"))
        self.lbl_crop_top = wx.StaticText(self.grp_processor, label=T("Top"))
        self.txt_crop_top = IntCtrl(
            self.grp_processor, size=self.FromDIP((60, -1)), allow_none=False, min=0, max=500, name="txt_crop_top"
        )
        self.txt_crop_top.SetValue(0)

        self.lbl_crop_left = wx.StaticText(self.grp_processor, label=T("Left"))
        self.txt_crop_left = IntCtrl(
            self.grp_processor, size=self.FromDIP((60, -1)), allow_none=False, min=0, max=500, name="txt_crop_left"
        )
        self.txt_crop_left.SetValue(0)

        self.lbl_crop_right = wx.StaticText(self.grp_processor, label=T("Right"))
        self.txt_crop_right = IntCtrl(
            self.grp_processor, size=self.FromDIP((60, -1)), allow_none=False, min=0, max=500, name="txt_crop_right"
        )
        self.txt_crop_right.SetValue(0)

        self.lbl_crop_bottom = wx.StaticText(self.grp_processor, label=T("Bottom"))
        self.txt_crop_bottom = IntCtrl(
            self.grp_processor, size=self.FromDIP((60, -1)), allow_none=False, min=0, max=500, name="txt_crop_bottom"
        )
        self.txt_crop_bottom.SetValue(0)

        self.chk_disable_draw_cursor = wx.CheckBox(
            self.grp_processor, label=T("Disable Draw Cursor"), name="chk_disable_draw_cursor"
        )
        self.chk_disable_draw_cursor.SetValue(False)

        self.chk_compile = wx.CheckBox(self.grp_processor, label=T("torch.compile"), name="chk_compile")
        self.chk_compile.SetToolTip(T("Enable model compiling"))
        self.chk_compile.SetValue(False)

        processor_layout = wx.GridBagSizer(vgap=5, hgap=4)
        processor_layout.SetEmptyCellSize((0, 0))
        processor_layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_device, (0, 1), flag=wx.EXPAND)
        processor_layout.Add(self.lbl_screenshot, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_screenshot, (1, 1), flag=wx.EXPAND)
        processor_layout.Add(self.chk_disable_draw_cursor, (2, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.lbl_monitor_index, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_monitor_index, (3, 1), flag=wx.EXPAND)
        processor_layout.Add(self.lbl_window_name, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_window_name, (4, 1), flag=wx.EXPAND)
        processor_layout.Add(self.btn_reload_window_name, (4, 2), flag=wx.EXPAND)
        processor_layout.Add(self.lbl_crop_region, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        crop_layout = wx.GridBagSizer(vgap=2, hgap=4)
        crop_layout.Add(self.lbl_crop_top, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_top, (0, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_left, (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_left, (0, 3), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_right, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_right, (1, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_bottom, (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_bottom, (1, 3), flag=wx.EXPAND)
        processor_layout.Add(crop_layout, (5, 1), flag=wx.EXPAND)

        processor_layout.Add(self.chk_compile, (6, 0), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        self.sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        self.sizer_processor.Add(processor_layout, 1, wx.ALL | wx.EXPAND, 4)
        return self.sizer_processor

    def _bind_desktop_events(self) -> None:
        self.cbo_divergence.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_synthetic_view.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_method.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_method)
        self.lbl_divergence_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_divergence_warning)
        self.chk_small_model_only.Bind(wx.EVT_CHECKBOX, self.update_depth_model_list)
        self.cbo_edge_dilation_y.Bind(wx.EVT_TEXT, self.on_changed_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)
        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.update_stereo_format)
        self.cbo_screenshot.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_screenshot)
        self.btn_reload_window_name.Bind(wx.EVT_BUTTON, self.update_window_names)
        self.cbo_device.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_device)
        self.chk_compile.Bind(wx.EVT_CHECKBOX, self.update_compile)

    def get_depth_models(self, small_only: bool) -> list[str]:
        depth_models = self._BaseDepthModel.get_available_models(desktop=True, small_only=small_only)
        if ADD_DEBUG_METHOD:
            depth_models.append("NULL")
        return depth_models

    def get_desktop_editable_comboboxes(self) -> list[EditableComboBox]:
        return [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_resolution,
            self.cbo_edge_dilation,
            self.cbo_edge_dilation_y,
            self.cbo_ema_decay,
            self.cbo_foreground_scale,
            self.cbo_stream_fps,
            self.cbo_stream_height,
            self.cbo_stream_quality,
        ]

    def get_anaglyph_method(self) -> str | None:
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            return self.cbo_anaglyph_method.GetValue()
        return None

    def update_wayland_warning(self) -> bool:
        if self._has_obs or self._has_kwcapture or not self._is_wayland():
            return False
        print(f"Warning: {self._wayland_missing_msg}", file=sys.stderr)
        message = self._T(self._wayland_missing_msg)
        self.cbo_screenshot.SetToolTip(message)
        self.SetStatusText(message)
        return True

    def update_depth_model_list(self, *args: Any, **kwargs: Any) -> None:
        default_model = "Any_V2_S"
        small_only = self.chk_small_model_only.IsChecked()
        depth_model = self.cbo_depth_model.GetValue()
        choices = self.get_depth_models(small_only=small_only)
        self.cbo_depth_model.SetItems(choices)

        if depth_model in choices:
            self.cbo_depth_model.SetSelection(choices.index(depth_model))
        else:
            self.cbo_depth_model.SetSelection(choices.index(default_model))

    def update_stereo_format(self, *args: Any, **kwargs: Any) -> None:
        stereo_format = self.cbo_stereo_format.GetValue()
        if stereo_format in {"Half SBS", "Half TB"}:
            self.lbl_format_device.SetLabel("Meta Quest 2/3")
        elif stereo_format in {"Full SBS", "Full TB"}:
            self.lbl_format_device.SetLabel("PICO 4")
        else:
            self.lbl_format_device.SetLabel("")
        self.update_anaglyph_state()

    def update_anaglyph_state(self) -> None:
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            self.lbl_anaglyph_method.Show()
            self.cbo_anaglyph_method.Show()
        else:
            self.lbl_anaglyph_method.Hide()
            self.cbo_anaglyph_method.Hide()
        self.GetSizer().Layout()

    def update_edge_dilation(self) -> None:
        if self.cbo_edge_dilation_y.GetValue():
            self.cbo_edge_dilation.SetToolTip(self._T("X"))
        else:
            self.cbo_edge_dilation.SetToolTip(self._T("X, Y"))

    def on_changed_edge_dilation(self, event: wx.Event) -> None:
        self.update_edge_dilation()

    def update_preserve_screen_border(self) -> None:
        if self.cbo_method.GetValue() in {
            "row_flow_v2",
            "row_flow_v3",
            "row_flow_v3_sym",
            "mlbw_l2",
            "mlbw_l4",
            "mlbw_l2s",
            "mlbw_l4s",
            "monobw",
        }:
            self.chk_preserve_screen_border.Enable()
        else:
            self.chk_preserve_screen_border.Disable()

    def on_selected_index_changed_cbo_method(self, event: wx.Event) -> None:
        self.update_divergence_warning()
        self.update_preserve_screen_border()

    def update_ema_normalize(self) -> None:
        if self.chk_ema_normalize.IsChecked():
            self.cbo_ema_decay.Enable()
        else:
            self.cbo_ema_decay.Disable()

    def on_changed_chk_ema_normalize(self, event: wx.Event) -> None:
        self.update_ema_normalize()

    def on_click_divergence_warning(self, event: wx.Event) -> None:
        self.lbl_divergence_warning.Hide()
        self.GetSizer().Layout()

    def update_divergence_warning(self, *args: Any, **kwargs: Any) -> None:
        T = self._T
        try:
            divergence = float(self.cbo_divergence.GetValue())
            method = self.cbo_method.GetValue()
            synthetic_view = self.cbo_synthetic_view.GetValue()
            max_divergence = float("inf")

            if method in {"row_flow_v3", "row_flow_v3_sym"}:
                max_divergence = 5.0 if synthetic_view == "both" else 5.0 * 0.5
            elif method == "row_flow_v2":
                max_divergence = 2.5 if synthetic_view == "both" else 2.5 * 0.5
            elif method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
                max_divergence = 10.0 if synthetic_view == "both" else 10.0 * 0.5

            if divergence > max_divergence:
                self.lbl_divergence_warning.SetLabel(
                    f"{divergence}: " + T("Out of range of training data") + f": {method}, {synthetic_view}"
                )
                self.lbl_divergence_warning.SetToolTip(T("This result could be unstable"))
                self.lbl_divergence_warning.Show()
            else:
                self.lbl_divergence_warning.SetLabel("")
                self.lbl_divergence_warning.SetToolTip("")
                self.lbl_divergence_warning.Hide()

            self.GetSizer().Layout()
        except ValueError:
            pass

    def on_selected_index_changed_cbo_device(self, event: wx.Event) -> None:
        self.update_compile()

    def update_compile(self, *args: Any, **kwargs: Any) -> None:
        selection = self.cbo_device.GetSelection()
        if selection == wx.NOT_FOUND:
            return
        device_id = int(self.cbo_device.GetClientData(selection))
        if device_id == -2:
            self.chk_compile.SetValue(False)
        elif self.chk_compile.IsChecked():
            device = self._create_device(device_id)
            if not self._check_compile_support(device):
                self.chk_compile.SetValue(False)

    def on_text_changed_cbo_screenshot(self, event: wx.Event) -> None:
        self.update_monitor_index()
        self.update_window_names()
        if self.IsShown():
            refresh_layouts(self)

    def update_monitor_index(self, *args: Any, **kwargs: Any) -> None:
        backend = self.cbo_screenshot.GetValue()
        if backend in BACKENDS_WITH_TARGET:
            self.lbl_monitor_index.Show()
            self.cbo_monitor_index.Show()
        else:
            self.lbl_monitor_index.Hide()
            self.cbo_monitor_index.Hide()
        self.chk_disable_draw_cursor.Enable(is_draw_cursor_supported(backend))
        self.GetSizer().Layout()

    def update_window_names(self, *args: Any, **kwargs: Any) -> None:
        backend = self.cbo_screenshot.GetValue()
        if backend in BACKENDS_WITH_TARGET:
            self.lbl_window_name.Show()
            self.cbo_window_name.Show()
            self.btn_reload_window_name.Show()
            self.cbo_window_name.SetItems([""] + self._enum_window_names())
        else:
            self.lbl_window_name.Hide()
            self.cbo_window_name.Hide()
            self.btn_reload_window_name.Hide()

        crop_widgets = (
            self.lbl_crop_region,
            self.txt_crop_top,
            self.lbl_crop_top,
            self.txt_crop_left,
            self.lbl_crop_left,
            self.txt_crop_right,
            self.lbl_crop_right,
            self.txt_crop_bottom,
            self.lbl_crop_bottom,
        )
        show_crop = backend in BACKENDS_WITH_CROP
        for w in crop_widgets:
            w.Show(show_crop)

        self.GetSizer().Layout()

    def show_validation_error_message(self, name: str, min_value: float, max_value: float) -> None:
        T = self._T
        wx.MessageBox(
            T("`{}` must be a number {} - {}").format(name, min_value, max_value),
            T("Error"),
            wx.OK | wx.ICON_ERROR,
        )

    def validate_desktop_stream_inputs(self, validate_port: bool = True) -> bool:
        T = self._T
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return False
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return False
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return False
        if not validate_number(self.cbo_edge_dilation_y.GetValue(), 0, 20, is_int=True, allow_empty=True):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return False
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return False
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return False
        if not validate_number(self.cbo_stream_fps.GetValue(), 1, 240, allow_empty=False):
            self.show_validation_error_message(T("Streaming FPS"), 1, 240)
            return False
        if not validate_number(self.cbo_stream_height.GetValue(), 320, 4320, allow_empty=False):
            self.show_validation_error_message(T("Streaming Resolution"), 320, 4320)
            return False
        if not validate_number(self.cbo_stream_quality.GetValue(), 1, 100, allow_empty=False):
            self.show_validation_error_message(T("MJPEG Quality"), 1, 100)
            return False
        if validate_port and not validate_number(self.txt_port.GetValue(), 1025, 65535, allow_empty=False):
            self.show_validation_error_message(T("Port"), 1025, 65535)
            return False

        if self.cbo_screenshot.GetValue() in BACKENDS_WITH_CROP:
            for crop_name, crop_value in [
                ("Top", self.txt_crop_top.GetValue()),
                ("Left", self.txt_crop_left.GetValue()),
                ("Right", self.txt_crop_right.GetValue()),
                ("Bottom", self.txt_crop_bottom.GetValue()),
            ]:
                if not validate_number(str(crop_value), 0, 500, is_int=True, allow_empty=False):
                    self.show_validation_error_message(T("Crop") + " " + T(crop_name), 0, 500)
                    return False

        resolution = self.cbo_resolution.GetValue()
        if resolution not in {"Default", ""}:
            if not validate_number(resolution, 192, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 192, 8190)
                return False

        return True

    def collect_desktop_stream_defaults(self) -> dict[str, Any]:
        resolution_str = self.cbo_resolution.GetValue()
        resolution = None if resolution_str in {"Default", ""} else int(resolution_str)

        selection = self.cbo_device.GetSelection()
        if selection == wx.NOT_FOUND:
            selection = 0
        device_id = [int(self.cbo_device.GetClientData(selection))]
        depth_model_type = self.cbo_depth_model.GetValue()
        if self.depth_model is None or (
            self.depth_model_type != depth_model_type
            or self.depth_model_device_id != device_id
            or self.depth_model_height != resolution
        ):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            self._gc_collect()

        if self.cbo_edge_dilation_y.GetValue():
            edge_dilation: int | list[int] = [
                int(self.cbo_edge_dilation.GetValue()),
                int(self.cbo_edge_dilation_y.GetValue()),
            ]
        else:
            edge_dilation = int(self.cbo_edge_dilation.GetValue())

        if self.cbo_screenshot.GetValue() in BACKENDS_WITH_CROP:
            crop_top = self.txt_crop_top.GetValue()
            crop_left = self.txt_crop_left.GetValue()
            crop_right = self.txt_crop_right.GetValue()
            crop_bottom = self.txt_crop_bottom.GetValue()
        else:
            crop_top = crop_left = crop_right = crop_bottom = 0

        monitor_index = int(self.cbo_monitor_index.GetValue() or 0)
        window_name = self.cbo_window_name.GetValue() or None
        if self.cbo_screenshot.GetValue() not in BACKENDS_WITH_TARGET:
            monitor_index = 0
            window_name = None

        stereo_format = self.cbo_stereo_format.GetValue()
        return {
            "gpu": device_id,
            "divergence": float(self.cbo_divergence.GetValue()),
            "convergence": float(self.cbo_convergence.GetValue()),
            "convergence_mode": self.cbo_convergence_mode.GetValue(),
            "synthetic_view": self.cbo_synthetic_view.GetValue(),
            "method": self.cbo_method.GetValue(),
            "preserve_screen_border": (
                self.chk_preserve_screen_border.IsEnabled() and self.chk_preserve_screen_border.IsChecked()
            ),
            "depth_model": depth_model_type,
            "foreground_scale": float(self.cbo_foreground_scale.GetValue()),
            "edge_dilation": edge_dilation,
            "ema_normalize": self.chk_ema_normalize.GetValue(),
            "ema_decay": float(self.cbo_ema_decay.GetValue()),
            "resolution": resolution,
            "autocrop": self.cbo_autocrop.GetValue() or None,
            "compile": self.chk_compile.IsEnabled() and self.chk_compile.IsChecked(),
            "screenshot": self.cbo_screenshot.GetValue(),
            "monitor_index": monitor_index,
            "window_name": window_name,
            "crop_top": crop_top,
            "crop_left": crop_left,
            "crop_right": crop_right,
            "crop_bottom": crop_bottom,
            "disable_draw_cursor": (
                self.chk_disable_draw_cursor.IsEnabled() and self.chk_disable_draw_cursor.IsChecked()
            ),
            "full_sbs": stereo_format == "Full SBS",
            "tb": stereo_format == "Full TB",
            "half_tb": stereo_format == "Half TB",
            "cross_eyed": stereo_format == "Cross Eyed",
            "rgbd": stereo_format == "RGB-D",
            "half_rgbd": stereo_format == "Half RGB-D",
            "anaglyph": self.get_anaglyph_method(),
            "pad_mode": "16:9" if self.chk_pad_16_9.IsChecked() else None,
            "stream_fps": int(self.cbo_stream_fps.GetValue()),
            "stream_height": int(self.cbo_stream_height.GetValue()),
            "stream_quality": int(self.cbo_stream_quality.GetValue()),
            "gpu_jpeg": self.chk_gpu_jpeg.IsEnabled() and self.chk_gpu_jpeg.IsChecked(),
        }

    def sync_depth_model_from_args(self, args: Any) -> None:
        if args is not None and args.state.get("depth_model") is not None and args.state["depth_model"].loaded():
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.resolution
