import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import os
import sys
import threading
import time
import traceback
from os import path

import wx
import wx.lib.agw.persist as persist
import wx.lib.stattext as stattext
from wx.lib.buttons import GenBitmapButton
from wx.lib.delayedresult import startWorker
from wx.lib.intctrl import IntCtrl

from iw3.locales import LOCALES, load_language_setting, save_language_setting
from nunif.gui import (
    EditableComboBox,
    EditableComboBoxPersistentHandler,
    IpAddrCtrl,
    apply_dark_mode,
    get_default_locale,
    init_win32_dpi,
    is_dark_mode,
    load_icon,
    persistent_manager_register,
    persistent_manager_register_all,
    persistent_manager_restore_all,
    persistent_manager_unregister_all,
    refresh_layouts,
    set_icon_ex,
    start_file,
    validate_number,
)
from nunif.utils.git import get_current_branch

from .dir_config import EXAMPLE_DIR, TMP_DIR
from .server import (
    create_parser,
    get_local_address,
    is_loopback_address,
    is_private_address,
    set_state_args,
)
from .server import (
    server_main as backend_main,
)

CONFIG_DIR = TMP_DIR
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-player-gui.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-player-lang.cfg")
DEFAULT_ROOT_DIR = EXAMPLE_DIR
os.makedirs(CONFIG_DIR, exist_ok=True)

BACKENDS_WITH_TARGET = {"wc_mp", "wc_cuda", "mss", "kwcapture"}
BACKENDS_WITH_CROP = BACKENDS_WITH_TARGET | {"obs"}
ADD_DEBUG_METHOD = bool(os.getenv("DEBUG", False))
BASIC_PERSIST_NAMES = {
    "iw3-player",
    "txt_url",
    "cbo_language",
    "txt_root",
    "chk_bind_addr",
    "txt_bind_addr",
    "txt_port",
    "chk_auth_enable",
    "txt_user",
    "txt_password",
    "chk_enable_stream",
}

myEVT_FPS = wx.NewEventType()
EVT_FPS = wx.PyEventBinder(myEVT_FPS, 0)


class FPSEvent(wx.PyCommandEvent):
    def __init__(
        self, etype, eid, estimated_fps=None, screenshot_fps=None, streaming_fps=None, screen_size=None, url=None
    ):
        super().__init__(etype, eid)
        self.estimated_fps = estimated_fps
        self.screenshot_fps = screenshot_fps
        self.streaming_fps = streaming_fps
        self.screen_size = screen_size
        self.url = url

    def GetValue(self):
        return (self.estimated_fps, self.screenshot_fps, self.streaming_fps, self.screen_size, self.url)


class FPSGUI:
    def __init__(self, parent, **kwargs):
        self.parent = parent

    def set_url(self, url):
        wx.PostEvent(self.parent, FPSEvent(myEVT_FPS, -1, None, None, None, None, url))

    def update(self, estimated_fps, screenshot_fps, streaming_fps, screen_size):
        wx.PostEvent(
            self.parent, FPSEvent(myEVT_FPS, -1, estimated_fps, screenshot_fps, streaming_fps, screen_size, None)
        )


class IW3PlayerApp(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker("iw3-player.lock", CONFIG_DIR)
        if self.instance.IsAnotherRunning():
            wx.MessageBox("Another instance is running", "Error", wx.OK | wx.ICON_ERROR)
            return False

        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
        refresh_layouts(main_frame)
        self.SetTopWindow(main_frame)

        return True


class MainFrame(wx.Frame):
    def __init__(self):
        branch_name = get_current_branch()
        if branch_name is None or branch_name in {"master", "main"}:
            branch_tag = ""
        else:
            branch_tag = f" ({branch_name})"

        python_version_tag = f" ({sys.implementation.name}-{sys.version_info[0]}.{sys.version_info[1]})"

        super().__init__(
            None,
            name="iw3-player",
            title=T("iw3-player") + branch_tag + python_version_tag,
            size=(480, 420),
            style=wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX,
        )
        self.stop_event = threading.Event()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self._desktop_ui_initialized = False
        self._enable_gpu_jpeg = False
        self._read_runtime_adjustment_values = None
        self._last_synced_stream_settings = None
        self.network_stream_widgets: list[wx.Window] = []
        self.processing = False
        self.args = None
        self.args_lock = threading.Lock()
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

    def initialize_component(self):
        normal_font = wx.Font(10, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        self._warning_font = wx.Font(
            8, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL
        )
        self._warning_color = (0xCC, 0x33, 0x33)

        self.SetFont(normal_font)
        self.CreateStatusBar()

        # Language panel
        self.pnl_top = wx.Panel(self)
        self.lbl_language = wx.StaticText(self.pnl_top, label=T("Language"))
        self.cbo_language = wx.ComboBox(self.pnl_top, name="cbo_language")
        lang_selection = 0
        for i, lang in enumerate(LOCAL_LIST):
            t = LOCALES.get(lang)
            name = t.get("_NAME", "Undefined")
            self.cbo_language.Append(name, lang)
            if lang in LOCALE_DICT.get("_LOCALE", []):
                lang_selection = i
        self.cbo_language.SetSelection(lang_selection)

        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(self.lbl_language, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        top_sizer.Add(self.cbo_language, flag=wx.ALL, border=2)
        top_sizer.AddSpacer(8)
        self.pnl_top.SetSizer(top_sizer)

        # Media Directory & Desktop Stream Toggle
        self.pnl_file = wx.Panel(self)
        self.grp_root = wx.StaticBox(self.pnl_file, label=T("Media Directory"))
        self.pnl_root = wx.Panel(self.grp_root)
        self.txt_root = wx.TextCtrl(self.pnl_root, name="txt_root")
        self.txt_root.SetValue(DEFAULT_ROOT_DIR)
        self.btn_browse_root = GenBitmapButton(self.pnl_root, bitmap=load_icon("folder-open.png"))
        self.btn_browse_root.SetToolTip(T("Choose a directory"))

        root_grid = wx.GridBagSizer(vgap=4, hgap=4)
        root_grid.Add(self.txt_root, (0, 0), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        root_grid.Add(self.btn_browse_root, (0, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        root_grid.AddGrowableCol(0, 1)
        self.pnl_root.SetSizer(root_grid)

        root_sizer = wx.StaticBoxSizer(self.grp_root, wx.VERTICAL)
        root_sizer.Add(self.pnl_root, 1, wx.ALL | wx.EXPAND, 4)

        self.chk_enable_stream = wx.CheckBox(self.pnl_file, label=T("Desktop Stream"), name="chk_enable_stream")
        self.chk_enable_stream.SetValue(False)

        file_sizer = wx.BoxSizer(wx.VERTICAL)
        file_sizer.Add(root_sizer, 0, wx.ALL | wx.EXPAND, 4)
        file_sizer.Add(self.chk_enable_stream, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.pnl_file.SetSizer(file_sizer)

        # Options panel
        self.pnl_options = wx.Panel(self)

        # Network Settings (basic fields always visible; stream fields added lazily)
        self.grp_network = wx.StaticBox(self.pnl_options, label=T("Network"))
        self.chk_bind_addr = wx.CheckBox(self.grp_network, label=T("Address"), name="chk_bind_addr")
        self.txt_bind_addr = IpAddrCtrl(self.grp_network, size=self.FromDIP((200, -1)), name="txt_bind_addr")
        self.chk_bind_addr.SetValue(False)
        self.txt_bind_addr.SetValue("127.0.0.1")
        self.btn_detect_ip = GenBitmapButton(self.grp_network, bitmap=load_icon("view-refresh.png"))
        self.btn_detect_ip.SetToolTip(T("Detect"))

        self.lbl_bind_addr_warning = stattext.GenStaticText(self.grp_network, label="")
        self.lbl_bind_addr_warning.SetFont(self._warning_font)
        self.lbl_bind_addr_warning.SetForegroundColour(self._warning_color)
        self.lbl_bind_addr_warning.Hide()

        self.lbl_port = wx.StaticText(self.grp_network, label=T("Port"))
        self.txt_port = IntCtrl(
            self.grp_network, size=self.FromDIP((200, -1)), allow_none=False, min=1025, max=65535, name="txt_port"
        )
        self.txt_port.SetValue(1304)

        self.sep_network1 = wx.StaticLine(self.grp_network, style=wx.LI_HORIZONTAL)
        self.chk_auth = wx.CheckBox(self.grp_network, label=T("Basic Authentication"), name="chk_auth_enable")
        self.lbl_user = wx.StaticText(self.grp_network, label=T("Username"))
        self.txt_user = wx.TextCtrl(self.grp_network, name="txt_user")
        self.lbl_password = wx.StaticText(self.grp_network, label=T("Password"))
        self.txt_password = wx.TextCtrl(self.grp_network, style=wx.TE_PASSWORD, name="txt_password")

        self.network_grid = wx.GridBagSizer(vgap=5, hgap=4)
        self.network_grid.SetEmptyCellSize((0, 0))
        self.network_grid.Add(self.chk_bind_addr, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.txt_bind_addr, (0, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.btn_detect_ip, (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.lbl_bind_addr_warning, (1, 0), (1, 3), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.lbl_port, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.txt_port, (2, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.sep_network1, (3, 0), (1, 3), flag=wx.EXPAND | wx.ALL)
        self.network_grid.Add(self.chk_auth, (4, 0), (1, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.lbl_user, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.txt_user, (5, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.lbl_password, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.txt_password, (6, 1), flag=wx.EXPAND)

        self.sizer_network = wx.StaticBoxSizer(self.grp_network, wx.VERTICAL)
        self.sizer_network.Add(self.network_grid, 1, wx.ALL | wx.EXPAND, 4)

        self.options_layout = wx.GridBagSizer(vgap=0, hgap=0)
        self.options_layout.SetEmptyCellSize((0, 0))
        self.options_layout.Add(self.sizer_network, (0, 1), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(self.options_layout)

        # Process Controls
        self.pnl_process = wx.Panel(self)
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_stop = wx.Button(self.pnl_process, label=T("Shutdown"))
        self.btn_stop.Disable()
        self.txt_url = wx.TextCtrl(self.pnl_process, size=self.FromDIP((300, -1)), style=wx.TE_READONLY)
        self.btn_url = GenBitmapButton(self.pnl_process, bitmap=load_icon("go-next.png"))
        self.btn_url.Disable()
        self.btn_url.SetToolTip(T("Open in Browser"))

        process_layout = wx.GridBagSizer(vgap=5, hgap=4)
        process_layout.Add(self.btn_start, (0, 0), flag=wx.EXPAND)
        process_layout.Add(self.btn_stop, (0, 1), flag=wx.EXPAND)
        process_layout.Add(self.txt_url, (0, 2), flag=wx.EXPAND)
        process_layout.Add(self.btn_url, (0, 3), flag=wx.EXPAND)
        self.pnl_process.SetSizer(process_layout)

        main_layout = wx.GridBagSizer(vgap=0, hgap=0)
        main_layout.SetEmptyCellSize((0, 0))
        main_layout.Add(self.pnl_top, (0, 0), flag=wx.ALIGN_RIGHT | wx.TOP, border=0)
        main_layout.Add(self.pnl_file, (1, 0), flag=wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, border=8)
        main_layout.Add(self.pnl_options, (2, 0), flag=wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, border=8)
        main_layout.Add(self.pnl_process, (3, 0), flag=wx.ALL | wx.EXPAND, border=8)
        self.SetSizer(main_layout)

        # Fix Frame and Panel background colors are different in Windows
        self.SetBackgroundColour(self.pnl_options.GetBackgroundColour())

        # Bind basic events
        self.cbo_language.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_language)
        self.btn_browse_root.Bind(wx.EVT_BUTTON, self.on_browse_root)

        self.chk_bind_addr.Bind(wx.EVT_CHECKBOX, self.update_bind_addr_state)
        self.txt_bind_addr.Bind(wx.EVT_TEXT, self.update_bind_addr_warning)
        self.lbl_bind_addr_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_bind_addr_warning)
        self.btn_detect_ip.Bind(wx.EVT_BUTTON, self.on_detect_ip)
        self.chk_auth.Bind(wx.EVT_CHECKBOX, self.update_auth_state)
        self.chk_enable_stream.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_enable_stream)

        self.btn_url.Bind(wx.EVT_BUTTON, self.on_open_url)
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)

        self.Bind(EVT_FPS, self.on_fps)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Load basic settings and sync state (lazily initializes desktop UI only if enabled)
        self.load_settings()
        self.update_bind_addr_state()
        self.update_bind_addr_warning()
        self.update_auth_state()
        self.update_stream_enable_state()

        self.btn_start.SetFocus()

    def _init_desktop_ui(self):
        if self._desktop_ui_initialized:
            return

        import importlib.util

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
        self._is_wayland = is_wayland
        self._wayland_missing_msg = WAYLAND_CAPTURE_MISSING_MESSAGE
        self._create_device = create_device
        self._check_compile_support = check_compile_support
        self._gc_collect = gc_collect
        self._enable_gpu_jpeg = ENABLE_GPU_JPEG
        self._read_runtime_adjustment_values = read_runtime_adjustment_values
        self._has_obs = is_obs_supported()
        self._has_kwcapture = is_kwcapture_supported()

        has_windows_capture = bool(importlib.util.find_spec("windows_capture"))
        is_rocm = getattr(torch.version, "hip", None) is not None
        has_windows_capture_cuda = (
            bool(importlib.util.find_spec("wc_cuda")) and torch.cuda.is_available() and not is_rocm
        )

        # Add streaming options inside grp_network below password
        self.sep_network2 = wx.StaticLine(self.grp_network, style=wx.LI_HORIZONTAL)
        self.lbl_stream_fps = wx.StaticText(self.grp_network, label=T("Streaming FPS"))
        self.cbo_stream_fps = EditableComboBox(
            self.grp_network, choices=["60", "30", "24", "15", "8"], name="cbo_stream_fps"
        )
        self.cbo_stream_fps.SetSelection(1)

        self.lbl_stream_height = wx.StaticText(self.grp_network, label=T("Streaming Resolution"))
        self.cbo_stream_height = EditableComboBox(self.grp_network, choices=["1080", "720"], name="cbo_stream_height")
        self.cbo_stream_height.SetSelection(0)

        self.lbl_stream_quality = wx.StaticText(self.grp_network, label=T("MJPEG Quality"))
        self.cbo_stream_quality = EditableComboBox(
            self.grp_network, choices=["100", "95", "90", "85", "80"], name="cbo_stream_quality"
        )
        self.cbo_stream_quality.SetSelection(2)

        self.chk_gpu_jpeg = wx.CheckBox(self.grp_network, label=T("GPU JPEG"), name="chk_gpu_jpeg")
        self.chk_gpu_jpeg.SetValue(False)
        if not self._enable_gpu_jpeg:
            self.chk_gpu_jpeg.Disable()

        self.network_grid.Add(self.sep_network2, (7, 0), (1, 3), flag=wx.EXPAND | wx.ALL)
        self.network_grid.Add(self.lbl_stream_fps, (8, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.cbo_stream_fps, (8, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.lbl_stream_height, (9, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.cbo_stream_height, (9, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.lbl_stream_quality, (10, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_grid.Add(self.cbo_stream_quality, (10, 1), flag=wx.EXPAND)
        self.network_grid.Add(self.chk_gpu_jpeg, (11, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        self.network_stream_widgets = [
            self.sep_network2,
            self.lbl_stream_fps,
            self.cbo_stream_fps,
            self.lbl_stream_height,
            self.cbo_stream_height,
            self.lbl_stream_quality,
            self.cbo_stream_quality,
            self.chk_gpu_jpeg,
        ]

        # Stereo Generation
        self.grp_stereo = wx.StaticBox(self.pnl_options, label=T("Stereo Generation"))
        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = EditableComboBox(
            self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"], name="cbo_divergence"
        )
        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(5)

        self.lbl_divergence_warning = stattext.GenStaticText(self.grp_stereo, label="")
        self.lbl_divergence_warning.SetFont(self._warning_font)
        self.lbl_divergence_warning.SetForegroundColour(self._warning_color)
        self.lbl_divergence_warning.Hide()

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

        self.sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        self.sizer_stereo.Add(stereo_layout, 1, wx.ALL | wx.EXPAND, 4)

        # Processor Settings
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=self.FromDIP((200, -1)), name="cbo_device")
        self.cbo_device.SetEditable(False)
        if torch.cuda.is_available():
            for dev_idx in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(dev_idx).name
                self.cbo_device.Append(f"{dev_idx}:{device_name}", dev_idx)
        elif mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif xpu_is_available():
            for dev_idx in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(dev_idx)
                self.cbo_device.Append(f"{dev_idx}:{device_name}", dev_idx)

        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_screenshot = wx.StaticText(self.grp_processor, label=T("Screenshot"))
        screenshot_backends = ["pil"]
        if is_mss_supported():
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
            self.grp_processor, choices=[str(i) for i in range(len(get_monitor_size_list()))], name="cbo_monitor_index"
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
        processor_layout.Add(self.lbl_monitor_index, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_monitor_index, (2, 1), flag=wx.EXPAND)
        processor_layout.Add(self.lbl_window_name, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.cbo_window_name, (3, 1), flag=wx.EXPAND)
        processor_layout.Add(self.btn_reload_window_name, (3, 2), flag=wx.EXPAND)
        processor_layout.Add(self.lbl_crop_region, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        crop_layout = wx.GridBagSizer(vgap=2, hgap=4)
        crop_layout.Add(self.lbl_crop_top, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_top, (0, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_left, (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_left, (0, 3), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_right, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_right, (1, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_bottom, (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_bottom, (1, 3), flag=wx.EXPAND)
        processor_layout.Add(crop_layout, (4, 1), flag=wx.EXPAND)

        processor_layout.Add(self.chk_disable_draw_cursor, (5, 0), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        processor_layout.Add(self.chk_compile, (6, 0), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        self.sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        self.sizer_processor.Add(processor_layout, 1, wx.ALL | wx.EXPAND, 4)

        self.options_layout.Add(self.sizer_stereo, (0, 0), (2, 1), flag=wx.ALL | wx.EXPAND, border=4)
        self.options_layout.Add(self.sizer_processor, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)

        # Bind desktop events
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

        if is_dark_mode():
            apply_dark_mode(self.grp_stereo)
            apply_dark_mode(self.grp_processor)
            for widget in self.network_stream_widgets:
                apply_dark_mode(widget)

        self._desktop_ui_initialized = True

        # Restore persisted values for newly constructed desktop controls
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        persistent_manager_restore_all(manager, exclude_names=BASIC_PERSIST_NAMES)
        persistent_manager_unregister_all(manager)

        self.update_depth_model_list()
        self.update_stereo_format()
        self.update_edge_dilation()
        self.update_ema_normalize()
        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_monitor_index()
        self.update_window_names()
        self.update_compile()
        self.update_wayland_warning()

    def update_stream_enable_state(self):
        enabled = self.chk_enable_stream.IsChecked()
        if enabled:
            self._init_desktop_ui()
            self.grp_stereo.Show()
            self.grp_processor.Show()
            for widget in self.network_stream_widgets:
                widget.Show()
            self.update_anaglyph_state()
            self.update_monitor_index()
            self.update_window_names()
            self.update_divergence_warning()
            self.update_bind_addr_warning()
        elif self._desktop_ui_initialized:
            self.grp_stereo.Hide()
            self.grp_processor.Hide()
            for widget in self.network_stream_widgets:
                widget.Hide()

        if self.IsShown():
            refresh_layouts(self)

    def on_changed_chk_enable_stream(self, event):
        self.update_stream_enable_state()

    def get_depth_models(self, small_only):
        depth_models = self._BaseDepthModel.get_available_models(desktop=True, small_only=small_only)
        if ADD_DEBUG_METHOD:
            depth_models.append("NULL")
        return depth_models

    def get_editable_comboboxes(self):
        if not self._desktop_ui_initialized:
            return []
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

    def get_anaglyph_method(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            return self.cbo_anaglyph_method.GetValue()
        return None

    def update_wayland_warning(self):
        if self._has_obs or self._has_kwcapture or not self._is_wayland():
            return False
        print(f"Warning: {self._wayland_missing_msg}", file=sys.stderr)
        message = T(self._wayland_missing_msg)
        self.cbo_screenshot.SetToolTip(message)
        self.SetStatusText(message)
        return True

    def on_close(self, event):
        self.save_settings()
        event.Skip()

        self.stop_event.set()
        if self.processing:
            max_wait = int(6 / 0.01)
            for _ in range(max_wait):
                if not self.stop_event.is_set():
                    break
                time.sleep(0.01)
            if self.stop_event.is_set():
                os._exit(-1)

    def on_browse_root(self, event):
        default_dir = self.txt_root.GetValue() or os.getcwd()
        with wx.DirDialog(
            self, T("Media Directory"), defaultPath=default_dir, style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_root.SetValue(dlg.GetPath())

    def on_detect_ip(self, event):
        self.txt_bind_addr.SetValue(get_local_address())

    def update_bind_addr_state(self, *args, **kwargs):
        if not self.chk_bind_addr.IsChecked():
            self.txt_bind_addr.SetValue(get_local_address())
            self.txt_bind_addr.Disable()
        else:
            self.txt_bind_addr.Enable()

    def update_bind_addr_warning(self, *args, **kwargs):
        bind_addr = self.txt_bind_addr.GetAddress()
        if is_loopback_address(bind_addr):
            self.lbl_bind_addr_warning.SetLabel(f"{bind_addr} is only accessible from this PC.")
            self.lbl_bind_addr_warning.Show()
        elif not is_private_address(bind_addr):
            self.lbl_bind_addr_warning.SetLabel(f"{bind_addr} is not Local Area Network Address.")
            self.lbl_bind_addr_warning.Show()
        else:
            self.lbl_bind_addr_warning.SetLabel("")
            self.lbl_bind_addr_warning.Hide()

        self.GetSizer().Layout()
        if self.IsShown():
            refresh_layouts(self)

    def on_click_bind_addr_warning(self, event):
        self.lbl_bind_addr_warning.Hide()
        self.GetSizer().Layout()
        if self.IsShown():
            refresh_layouts(self)

    def update_auth_state(self, *args, **kwargs):
        if self.chk_auth.IsChecked():
            self.txt_user.Enable()
            self.txt_password.Enable()
        else:
            self.txt_user.Disable()
            self.txt_password.Disable()

    def update_depth_model_list(self, *args, **kwargs):
        default_model = "Any_V2_S"
        small_only = self.chk_small_model_only.IsChecked()
        depth_model = self.cbo_depth_model.GetValue()
        choices = self.get_depth_models(small_only=small_only)
        self.cbo_depth_model.SetItems(choices)

        if depth_model in choices:
            self.cbo_depth_model.SetSelection(choices.index(depth_model))
        else:
            self.cbo_depth_model.SetSelection(choices.index(default_model))

    def update_stereo_format(self, *args, **kwargs):
        stereo_format = self.cbo_stereo_format.GetValue()
        if stereo_format in {"Half SBS", "Half TB"}:
            self.lbl_format_device.SetLabel("Meta Quest 2/3")
        elif stereo_format in {"Full SBS", "Full TB"}:
            self.lbl_format_device.SetLabel("PICO 4")
        else:
            self.lbl_format_device.SetLabel("")
        self.update_anaglyph_state()

    def update_anaglyph_state(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            self.lbl_anaglyph_method.Show()
            self.cbo_anaglyph_method.Show()
        else:
            self.lbl_anaglyph_method.Hide()
            self.cbo_anaglyph_method.Hide()
        self.GetSizer().Layout()
        if self.IsShown():
            refresh_layouts(self)

    def update_edge_dilation(self):
        if self.cbo_edge_dilation_y.GetValue():
            self.cbo_edge_dilation.SetToolTip(T("X"))
        else:
            self.cbo_edge_dilation.SetToolTip(T("X, Y"))

    def on_changed_edge_dilation(self, event):
        self.update_edge_dilation()

    def update_preserve_screen_border(self):
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

    def on_selected_index_changed_cbo_method(self, event):
        self.update_divergence_warning()
        self.update_preserve_screen_border()

    def update_ema_normalize(self):
        if self.chk_ema_normalize.IsChecked():
            self.cbo_ema_decay.Enable()
        else:
            self.cbo_ema_decay.Disable()

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()

    def on_click_divergence_warning(self, event):
        self.lbl_divergence_warning.Hide()
        self.GetSizer().Layout()
        if self.IsShown():
            refresh_layouts(self)

    def update_divergence_warning(self, *args, **kwargs):
        try:
            divergence = float(self.cbo_divergence.GetValue())
            method = self.cbo_method.GetValue()
            synthetic_view = self.cbo_synthetic_view.GetValue()
            max_divergence = float("inf")

            if method in {"row_flow_v3", "row_flow_v3_sym"}:
                max_divergence = 5.0 if synthetic_view == "both" else 2.5
            elif method == "row_flow_v2":
                max_divergence = 2.5 if synthetic_view == "both" else 1.25
            elif method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
                max_divergence = 10.0 if synthetic_view == "both" else 5.0

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
            if self.IsShown():
                refresh_layouts(self)
        except ValueError:
            pass

    def on_selected_index_changed_cbo_device(self, event):
        self.update_compile()

    def update_compile(self, *args, **kwargs):
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

    def on_text_changed_cbo_language(self, event):
        lang = self.cbo_language.GetClientData(self.cbo_language.GetSelection())
        save_language_setting(LANG_CONFIG_PATH, lang)
        with wx.MessageDialog(
            None, message=T("The language setting will be applied after restarting"), style=wx.OK
        ) as dlg:
            dlg.ShowModal()

    def on_text_changed_cbo_screenshot(self, event):
        self.update_monitor_index()
        self.update_window_names()
        if self.IsShown():
            refresh_layouts(self)

    def update_monitor_index(self, *args, **kwargs):
        if self.cbo_screenshot.GetValue() in BACKENDS_WITH_TARGET:
            self.lbl_monitor_index.Show()
            self.cbo_monitor_index.Show()
        else:
            self.lbl_monitor_index.Hide()
            self.cbo_monitor_index.Hide()

        self.GetSizer().Layout()

    def update_window_names(self, *args, **kwargs):
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

        crop_widgets = [
            self.lbl_crop_region,
            self.txt_crop_top,
            self.lbl_crop_top,
            self.txt_crop_left,
            self.lbl_crop_left,
            self.txt_crop_right,
            self.lbl_crop_right,
            self.txt_crop_bottom,
            self.lbl_crop_bottom,
        ]
        if backend in BACKENDS_WITH_CROP:
            for w in crop_widgets:
                w.Show()
        else:
            for w in crop_widgets:
                w.Hide()

        self.GetSizer().Layout()

    def on_open_url(self, event):
        url = self.txt_url.GetValue()
        if url:
            start_file(url)

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
            None,
            message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
            caption=T("Error"),
            style=wx.OK,
        ) as dlg:
            dlg.ShowModal()

    def parse_args(self):
        root = self.txt_root.GetValue()
        if not root or not path.isdir(root):
            wx.MessageBox(T("Please select a valid Media Directory."), T("Error"), wx.OK | wx.ICON_ERROR)
            return None
        if not validate_number(self.txt_port.GetValue(), 1025, 65535, allow_empty=False):
            self.show_validation_error_message(T("Port"), 1025, 65535)
            return None

        bind_addr = self.txt_bind_addr.GetAddress() if self.chk_bind_addr.IsChecked() else None
        if self.chk_auth.IsChecked():
            user = self.txt_user.GetValue()
            password = self.txt_password.GetValue()
        else:
            user = password = None

        enable_stream = self.chk_enable_stream.IsChecked()
        if not enable_stream:
            parser = create_parser(enable_stream=False)
            parser.set_defaults(
                root=root,
                bind_addr=bind_addr,
                port=int(self.txt_port.GetValue()),
                user=user,
                password=password,
                enable_stream=False,
            )
            args = parser.parse_args([])
            set_state_args(
                args,
                args_lock=self.args_lock,
                fps_event=FPSGUI(self),
                stop_event=self.stop_event,
            )
            return args

        self._init_desktop_ui()
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return None
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.cbo_edge_dilation_y.GetValue(), 0, 20, is_int=True, allow_empty=True):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return None
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return None
        if not validate_number(self.cbo_stream_fps.GetValue(), 1, 240, allow_empty=False):
            self.show_validation_error_message(T("Streaming FPS"), 1, 240)
            return None
        if not validate_number(self.cbo_stream_height.GetValue(), 320, 4320, allow_empty=False):
            self.show_validation_error_message(T("Streaming Resolution"), 320, 4320)
            return None
        if not validate_number(self.cbo_stream_quality.GetValue(), 1, 100, allow_empty=False):
            self.show_validation_error_message(T("MJPEG Quality"), 1, 100)
            return None

        if self.cbo_screenshot.GetValue() in BACKENDS_WITH_CROP:
            crop_top = self.txt_crop_top.GetValue()
            crop_left = self.txt_crop_left.GetValue()
            crop_right = self.txt_crop_right.GetValue()
            crop_bottom = self.txt_crop_bottom.GetValue()
            for crop_name, crop_value in [
                ("Top", crop_top),
                ("Left", crop_left),
                ("Right", crop_right),
                ("Bottom", crop_bottom),
            ]:
                if not validate_number(str(crop_value), 0, 500, is_int=True, allow_empty=False):
                    self.show_validation_error_message(T("Crop") + " " + T(crop_name), 0, 500)
                    return None
        else:
            crop_top = crop_left = crop_right = crop_bottom = 0

        resolution = self.cbo_resolution.GetValue()
        if resolution == "Default" or resolution == "":
            resolution = None
        else:
            if not validate_number(resolution, 192, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 192, 8190)
                return None
            resolution = int(resolution)

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
            self.depth_model_height = None
            self._gc_collect()

        if self.cbo_edge_dilation_y.GetValue():
            edge_dilation = [int(self.cbo_edge_dilation.GetValue()), int(self.cbo_edge_dilation_y.GetValue())]
        else:
            edge_dilation = int(self.cbo_edge_dilation.GetValue())

        preserve_screen_border = (
            self.chk_preserve_screen_border.IsEnabled() and self.chk_preserve_screen_border.IsChecked()
        )
        monitor_index = int(self.cbo_monitor_index.GetValue() or 0)
        window_name = self.cbo_window_name.GetValue()
        if self.cbo_screenshot.GetValue() not in BACKENDS_WITH_TARGET:
            monitor_index = 0
            window_name = None
        if not window_name:
            window_name = None

        stereo_format = self.cbo_stereo_format.GetValue()
        full_sbs = stereo_format == "Full SBS"
        tb = stereo_format == "Full TB"
        half_tb = stereo_format == "Half TB"
        cross_eyed = stereo_format == "Cross Eyed"
        rgbd = stereo_format == "RGB-D"
        half_rgbd = stereo_format == "Half RGB-D"
        anaglyph = self.get_anaglyph_method()
        pad_mode = "16:9" if self.chk_pad_16_9.IsChecked() else None

        parser = create_parser(enable_stream=True)
        parser.set_defaults(
            root=root,
            bind_addr=bind_addr,
            port=int(self.txt_port.GetValue()),
            user=user,
            password=password,
            enable_stream=True,
            stream_fps=int(self.cbo_stream_fps.GetValue()),
            stream_height=int(self.cbo_stream_height.GetValue()),
            stream_quality=int(self.cbo_stream_quality.GetValue()),
            gpu_jpeg=self.chk_gpu_jpeg.IsEnabled() and self.chk_gpu_jpeg.IsChecked(),
            gpu=device_id,
            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            convergence_mode=self.cbo_convergence_mode.GetValue(),
            synthetic_view=self.cbo_synthetic_view.GetValue(),
            method=self.cbo_method.GetValue(),
            preserve_screen_border=preserve_screen_border,
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            edge_dilation=edge_dilation,
            ema_normalize=self.chk_ema_normalize.GetValue(),
            ema_decay=float(self.cbo_ema_decay.GetValue()),
            resolution=resolution,
            autocrop=self.cbo_autocrop.GetValue() if self.cbo_autocrop.GetValue() else None,
            compile=self.chk_compile.IsEnabled() and self.chk_compile.IsChecked(),
            screenshot=self.cbo_screenshot.GetValue(),
            monitor_index=monitor_index,
            window_name=window_name,
            crop_top=crop_top,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
            disable_draw_cursor=self.chk_disable_draw_cursor.GetValue(),
            full_sbs=full_sbs,
            tb=tb,
            half_tb=half_tb,
            cross_eyed=cross_eyed,
            rgbd=rgbd,
            half_rgbd=half_rgbd,
            anaglyph=anaglyph,
            pad_mode=pad_mode,
        )
        args = parser.parse_args([])
        set_state_args(
            args,
            args_lock=self.args_lock,
            fps_event=FPSGUI(self),
            stop_event=self.stop_event,
            depth_model=self.depth_model,
        )
        return args

    def _sync_stream_settings_to_gui(self, args):
        if (
            not self._desktop_ui_initialized
            or self._read_runtime_adjustment_values is None
            or args is None
            or not getattr(args, "enable_stream", False)
        ):
            return
        current = self._read_runtime_adjustment_values(args, self.args_lock)
        if current is None:
            return
        prev = self._last_synced_stream_settings
        if current == prev:
            return
        div, conv, fg, edge_x, edge_y = current
        if prev is None or div != prev[0]:
            self.cbo_divergence.SetValue(f"{div:.2f}".rstrip("0").rstrip("."))
        if prev is None or conv != prev[1]:
            self.cbo_convergence.SetValue(f"{conv:.2f}".rstrip("0").rstrip("."))
        if prev is None or fg != prev[2]:
            self.cbo_foreground_scale.SetValue(f"{fg:.2f}".rstrip("0").rstrip("."))
        if prev is None or edge_x != prev[3]:
            self.cbo_edge_dilation.SetValue(str(edge_x))
        if prev is None or edge_y != prev[4]:
            self.cbo_edge_dilation_y.SetValue(edge_y)
        self._last_synced_stream_settings = current

    def on_fps(self, event):
        estimated_fps, screenshot_fps, streaming_fps, screen_size, url = event.GetValue()
        if self.args is not None:
            self._sync_stream_settings_to_gui(self.args)
        if url:
            self.txt_url.SetValue(url)
            self.btn_url.Enable()
        if (
            screen_size is not None
            and estimated_fps is not None
            and screenshot_fps is not None
            and streaming_fps is not None
        ):
            self.SetStatusText(
                f"Estimated FPS: {estimated_fps:.02f},"
                f" Screenshot FPS: {screenshot_fps:.02f},"
                f" Streaming FPS: {streaming_fps:.02f},"
                f" Screen: {screen_size}"
            )
        elif screen_size is None and url is None and self.processing:
            self.SetStatusText("Ready (Desktop Stream: Idle)")

    def on_start(self, event):
        self.args = self.parse_args()
        if self.args is None:
            return

        if (
            self._desktop_ui_initialized
            and self._read_runtime_adjustment_values is not None
            and getattr(self.args, "enable_stream", False)
        ):
            self._last_synced_stream_settings = self._read_runtime_adjustment_values(self.args)
        else:
            self._last_synced_stream_settings = None

        self.stop_event.clear()
        self.btn_start.Disable()
        self.btn_stop.Enable()
        self.pnl_top.Disable()
        self.pnl_file.Disable()
        self.pnl_options.Disable()

        if self.args.bind_addr:
            url = f"https://{self.args.bind_addr}:{self.args.port}"
            self.txt_url.SetValue(url)
            self.btn_url.Enable()

        self.SetStatusText(T("Running"))
        self.processing = True
        startWorker(self.on_exit_worker, backend_main, wargs=(self.args, False))

    def on_stop(self, event):
        self.stop_event.set()
        self.btn_stop.Disable()

    def on_exit_worker(self, result):
        error_info = None
        final_args = None
        try:
            args = result.get()
            final_args = args
            if (
                self._desktop_ui_initialized
                and args is not None
                and args.state.get("depth_model") is not None
                and args.state["depth_model"].loaded()
            ):
                self.depth_model = args.state["depth_model"]
                self.depth_model_type = args.depth_model
                self.depth_model_device_id = args.gpu
                self.depth_model_height = args.resolution
            self.SetStatusText(T("Shutdown"))
        except Exception:
            final_args = self.args
            if (
                self._desktop_ui_initialized
                and self.args is not None
                and self.args.state.get("depth_model") is not None
                and self.args.state["depth_model"].loaded()
            ):
                self.depth_model = self.args.state["depth_model"]
                self.depth_model_type = self.args.depth_model
                self.depth_model_device_id = self.args.gpu
                self.depth_model_height = self.args.resolution
            self.SetStatusText(T("Error"))
            _e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            print(e, file=sys.stderr)
            traceback.print_tb(tb)
            error_info = (message, f"{T('Error')}: {e.__class__.__name__}")

        if final_args is not None:
            self._sync_stream_settings_to_gui(final_args)
        self._last_synced_stream_settings = None
        self.args = None
        self.processing = False
        self.btn_start.Enable()
        self.btn_stop.Disable()
        self.txt_url.SetValue("")
        self.btn_url.Disable()
        self.pnl_top.Enable()
        self.pnl_file.Enable()
        self.pnl_options.Enable()

        self.update_bind_addr_state()
        self.update_auth_state()
        if self._desktop_ui_initialized:
            self.update_ema_normalize()
            self.update_preserve_screen_border()
            if not self._enable_gpu_jpeg:
                self.chk_gpu_jpeg.Disable()

        refresh_layouts(self)
        wx.CallAfter(refresh_layouts, self)

        if error_info is not None:
            wx.MessageBox(error_info[0], error_info[1], wx.OK | wx.ICON_ERROR)
            refresh_layouts(self)
            wx.CallAfter(refresh_layouts, self)

        if self._desktop_ui_initialized:
            self._gc_collect()

    def load_settings(self):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        persistent_manager_restore_all(manager, exclude_names=["txt_url", "cbo_language"])
        persistent_manager_unregister_all(manager)

    def save_settings(self):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        manager.SaveAndUnregister()


LOCAL_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse

    global LOCALE_DICT

    try:
        from .download_assets import main as download_main

        download_main()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, choices=list(LOCALES.keys()), help="translation language")
    args = parser.parse_args()
    if args.lang:
        LOCALE_DICT = LOCALES.get(args.lang, {})
    else:
        saved_lang = load_language_setting(LANG_CONFIG_PATH)
        if saved_lang:
            LOCALE_DICT = LOCALES.get(saved_lang, {})

    sys.argv = [sys.argv[0]]
    app = IW3PlayerApp()
    app.MainLoop()


if __name__ == "__main__":
    init_win32_dpi()
    main()
