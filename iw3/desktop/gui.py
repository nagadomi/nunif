import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import functools
import math
import os
import sys
import threading
import time
import traceback
from os import path
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as persist
import wx.lib.stattext as stattext
from wx.lib.buttons import GenBitmapButton
from wx.lib.intctrl import IntCtrl
from nunif.utils.git import get_current_branch
from nunif.initializer import gc_collect
from nunif.utils.filename import sanitize_filename
from nunif.utils.home_dir import ensure_home_dir
from nunif.gui import (
    EVT_TQDM,
    TQDMGUI,
    EditableComboBox,
    EditableComboBoxPersistentHandler,
    FPSTracker,
    IpAddrCtrl,
    persistent_manager_register_all,
    persistent_manager_unregister_all,
    persistent_manager_restore_all,
    persistent_manager_register,
    set_icon_ex,
    load_icon,
    start_file,
    apply_dark_mode,
    is_dark_mode,
    get_default_locale,
    refresh_layouts,
)
from ..locales import LOCALES, load_language_setting, save_language_setting
from ..player.server import get_local_address, is_loopback_address, is_private_address
from .gui_shared import (
    EVT_FPS,
    FPSGUI,
    DesktopStreamGUIMixin,
)
from .utils import (
    init_win32,
    iw3_desktop_main,
    create_parser,
    set_state_args,
    read_runtime_adjustment_values,
    IW3U,
)


CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "..", "tmp"))
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-desktop.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-desktop-lang.cfg")
PRESET_DIR = path.join(CONFIG_DIR, "presets")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

LAYOUT_DEBUG = False


# Spin inclrement value
DIVERGENCE_INC_DEFAULT = 0.25
DIVERGENCE_INC_FINE = 0.005
CONVERGENCE_INC_DEFAULT = 0.1
CONVERGENCE_INC_FINE = 0.005
FOREGROUNDSCALE_INC_DEFAULT = 0.2
FOREGROUNDSCALE_INC_FINE = 0.005


class IW3DesktopApp(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker("iw3-desktop-gui.lock", CONFIG_DIR)
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(
                None, message=T("Another instance is running"), caption=T("Confirm"), style=wx.OK
            ) as dlg:
                dlg.ShowModal()
                return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
        refresh_layouts(main_frame)
        self.SetTopWindow(main_frame)

        return True


class MainFrame(wx.Frame, DesktopStreamGUIMixin):
    def __init__(self):
        branch_name = get_current_branch()
        if branch_name is None or branch_name in {"master", "main"}:
            branch_tag = ""
        else:
            branch_tag = f" ({branch_name})"

        python_version_tag = f" ({sys.implementation.name}-{sys.version_info[0]}.{sys.version_info[1]})"

        super(MainFrame, self).__init__(
            None,
            name="iw3-desktop",
            title=T("iw3-desktop") + branch_tag + python_version_tag,
            size=(720, 560),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX),
        )
        self.stop_event = threading.Event()
        self.fps_tracker = FPSTracker(update_interval=0.5)
        self._tqdm_pos = 0
        self._tqdm_total = 1
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.processing = False
        self.args = None
        self.args_lock = threading.Lock()
        self._last_synced_adj = None
        self._last_webxr_video_codec = None
        self._last_legacy_video_codec = None
        self._init_desktop_deps()
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

    def _T(self, s: str) -> str:
        return T(s)

    def initialize_component(self):
        NORMAL_FONT = wx.Font(10, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_FONT = wx.Font(8, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_COLOR = (0xCC, 0x33, 0x33)

        self.SetFont(NORMAL_FONT)
        self.CreateStatusBar()

        # options panel
        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # stereo generation settings
        sizer_stereo = self._build_stereo_group(self.pnl_options, WARNING_FONT, WARNING_COLOR)
        if LAYOUT_DEBUG:
            self.grp_stereo.SetBackgroundColour("#fcf")

        # network settings
        self.grp_network = wx.StaticBox(self.pnl_options, label=T("Network"))
        if LAYOUT_DEBUG:
            self.grp_network.SetBackgroundColour("#fcf")

        self.lbl_view_mode = wx.StaticText(self.grp_network, label=T("View Mode"))
        self.cbo_view_mode = wx.ComboBox(
            self.grp_network, choices=["Web Streaming", "Web Streaming (Legacy)", "Local Viewer"], name="cbo_view_mode"
        )
        self.cbo_view_mode.SetEditable(False)
        self.cbo_view_mode.SetSelection(0)

        self.chk_bind_addr = wx.CheckBox(self.grp_network, label=T("Address"), name="chk_bind_addr")
        self.txt_bind_addr = IpAddrCtrl(self.grp_network, size=self.FromDIP((200, -1)), name="txt_bind_addr")
        self.chk_bind_addr.SetValue(False)
        self.txt_bind_addr.SetValue("127.0.0.1")
        self.btn_detect_ip = GenBitmapButton(self.grp_network, bitmap=load_icon("view-refresh.png"))
        self.btn_detect_ip.SetToolTip(T("Detect"))

        self.lbl_bind_addr_warning = stattext.GenStaticText(self.grp_network, label="")
        self.lbl_bind_addr_warning.SetFont(WARNING_FONT)
        self.lbl_bind_addr_warning.SetForegroundColour(WARNING_COLOR)
        self.lbl_bind_addr_warning.Hide()

        self.lbl_port = wx.StaticText(self.grp_network, label=T("Port"))
        self.txt_port = IntCtrl(
            self.grp_network, size=self.FromDIP((200, -1)), allow_none=False, min=1025, max=65535, name="txt_port"
        )
        self.txt_port.SetValue(1303)
        self.lbl_stream_fps = wx.StaticText(self.grp_network, label=T("Streaming FPS"))
        self.cbo_stream_fps = EditableComboBox(
            self.grp_network, choices=["60", "30", "24", "15", "8"], name="cbo_stream_fps"
        )
        self.cbo_stream_fps.SetSelection(1)

        self.chk_uncap_fps = wx.CheckBox(self.grp_network, label=T("Uncap FPS"), name="chk_uncap_fps")
        self.chk_uncap_fps.SetValue(False)

        self.lbl_stream_height = wx.StaticText(self.grp_network, label=T("Streaming Resolution"))
        self.cbo_stream_height = EditableComboBox(
            self.grp_network, choices=["2160", "1080", "720"], name="cbo_stream_height"
        )
        self.cbo_stream_height.SetSelection(1)

        self.lbl_stream_quality = wx.StaticText(self.grp_network, label=T("Streaming Quality"))
        self.cbo_stream_quality = EditableComboBox(
            self.grp_network, choices=["100", "95", "90", "85", "80"], name="cbo_stream_quality"
        )
        self.cbo_stream_quality.SetSelection(2)
        self.lbl_video_codec = wx.StaticText(self.grp_network, label=T("Video Codec"))
        self.cbo_video_codec = wx.ComboBox(
            self.grp_network,
            choices=list(self._available_stream_video_codecs),
            name="cbo_video_codec",
        )
        self.cbo_video_codec.SetEditable(False)
        self.cbo_video_codec.SetSelection(0)
        self.chk_pad_16_9 = wx.CheckBox(self.grp_network, label=T("Padding") + " 16:9", name="chk_pad_16_9")
        self.chk_pad_16_9.SetValue(False)
        self.chk_fit_window = wx.CheckBox(self.grp_network, label=T("Fit Window"), name="chk_fit_window")
        self.chk_fit_window.SetValue(False)

        self.sep_network1 = wx.StaticLine(self.grp_network, style=wx.LI_HORIZONTAL)

        self.chk_auth = wx.CheckBox(self.grp_network, label=T("Basic Authentication"), name="chk_auth")
        self.lbl_auth_username = wx.StaticText(self.grp_network, label=T("Username"))
        self.txt_auth_username = wx.TextCtrl(self.grp_network, name="txt_auth_username")
        self.lbl_auth_password = wx.StaticText(self.grp_network, label=T("Password"))
        self.txt_auth_password = wx.TextCtrl(self.grp_network, style=wx.TE_PASSWORD, name="txt_auth_password")

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.lbl_view_mode, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_view_mode, (0, 1), flag=wx.EXPAND)

        layout.Add(self.chk_bind_addr, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_bind_addr, (1, 1), flag=wx.EXPAND)
        layout.Add(self.btn_detect_ip, (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_bind_addr_warning, (2, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_port, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_port, (3, 1), flag=wx.EXPAND)

        layout.Add(self.lbl_stream_fps, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stream_fps, (4, 1), flag=wx.EXPAND)
        layout.Add(self.chk_uncap_fps, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_stream_height, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stream_height, (6, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_stream_quality, (7, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stream_quality, (7, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_codec, (8, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_codec, (8, 1), flag=wx.EXPAND)
        layout.Add(self.chk_pad_16_9, (9, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_fit_window, (10, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sep_network1, (11, 0), (0, 3), flag=wx.EXPAND | wx.ALL)

        layout.Add(self.chk_auth, (12, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_auth_username, (13, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_auth_username, (13, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_auth_password, (14, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_auth_password, (14, 1), flag=wx.EXPAND)

        sizer_network = wx.StaticBoxSizer(self.grp_network, wx.VERTICAL)
        sizer_network.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        sizer_processor = self._build_processor_group(self.pnl_options)
        if LAYOUT_DEBUG:
            self.grp_processor.SetBackgroundColour("#fcf")

        # adjustment settings
        self.grp_adjustment = wx.StaticBox(self.pnl_options, label=T("Adjustment"))
        if LAYOUT_DEBUG:
            self.grp_adjustment.SetBackgroundColour("#ccc")

        self.lbl_adj_divergence = wx.StaticText(self.grp_adjustment, label=T("3D Strength"))
        self.sld_adj_divergence = wx.SpinCtrlDouble(
            self.grp_adjustment, value="1.00", min=0.0, max=10.0, inc=DIVERGENCE_INC_DEFAULT
        )
        self.sld_adj_divergence.SetDigits(2)
        self.lbl_adj_convergence = wx.StaticText(self.grp_adjustment, label=T("Convergence Plane"))
        self.sld_adj_convergence = wx.SpinCtrlDouble(
            self.grp_adjustment, value="1.0", min=0.0, max=1.0, inc=CONVERGENCE_INC_DEFAULT
        )
        self.lbl_adj_foreground_scale = wx.StaticText(self.grp_adjustment, label=T("Foreground Scale"))
        self.sld_adj_foreground_scale = wx.SpinCtrlDouble(
            self.grp_adjustment, value="0", min=-3.0, max=3.0, inc=FOREGROUNDSCALE_INC_DEFAULT
        )
        self.lbl_adj_edge_dilation = wx.StaticText(self.grp_adjustment, label=T("Edge Fix"))
        self.sld_adj_edge_dilation = wx.SpinCtrl(
            self.grp_adjustment, value="0", min=0, max=10, name="sld_adj_edge_dilation"
        )
        self.cbo_adj_edge_dilation_y = EditableComboBox(
            self.grp_adjustment, choices=["", "0", "1", "2"], name="cbo_adj_edge_dilation_y"
        )
        self.tgl_adj_fine = wx.ToggleButton(self.grp_adjustment, label=T("Fine Adjustment"), name="tgl_adj_fine")
        self.btn_adj_align = wx.Button(self.grp_adjustment, label=T("Align to Step"))

        self.set_flat_size(self.tgl_adj_fine)
        self.set_flat_size(self.btn_adj_align)

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.lbl_adj_divergence, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_adj_divergence, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_adj_convergence, (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_adj_convergence, (0, 3), flag=wx.EXPAND)

        layout.Add(self.lbl_adj_foreground_scale, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_adj_foreground_scale, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_adj_edge_dilation, (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_adj_edge_dilation, (1, 3), flag=wx.EXPAND)
        layout.Add(self.cbo_adj_edge_dilation_y, (1, 4), flag=wx.EXPAND)
        layout.Add(self.tgl_adj_fine, (2, 0), flag=wx.EXPAND)
        layout.Add(self.btn_adj_align, (2, 1), flag=wx.EXPAND)

        sizer_adjustment = wx.StaticBoxSizer(self.grp_adjustment, wx.VERTICAL)
        sizer_adjustment.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # options

        layout = wx.GridBagSizer(vgap=0, hgap=0)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(sizer_adjustment, (0, 0), (0, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_stereo, (1, 0), (2, 0), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_network, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, (2, 1), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(layout)

        # processing panel
        self.pnl_process = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_process.SetBackgroundColour("#fcc")

        self.txt_url = wx.TextCtrl(self.pnl_process, size=self.FromDIP((300, -1)), style=wx.TE_READONLY)
        self.btn_url = GenBitmapButton(self.pnl_process, bitmap=load_icon("go-next.png"))
        self.btn_url.Disable()
        self.btn_url.SetToolTip(T("Open in Browser"))
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Shutdown"))

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.Add(self.btn_start, (0, 0), flag=wx.EXPAND)
        layout.Add(self.btn_cancel, (0, 1), flag=wx.EXPAND)
        layout.Add(self.txt_url, (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_url, (0, 3), flag=wx.EXPAND)
        self.pnl_process.SetSizer(layout)

        # language panel
        self.pnl_preset = wx.Panel(self)
        # language
        self.lbl_language = wx.StaticText(self.pnl_preset, label=T("Language"))
        self.cbo_language = wx.ComboBox(self.pnl_preset, name="cbo_language")
        lang_selection = 0
        for i, lang in enumerate(LOCALE_LIST):
            t = LOCALES.get(lang)
            name = t.get("_NAME", "Undefined")
            self.cbo_language.Append(name, lang)
            if lang in LOCALE_DICT.get("_LOCALE", []):
                lang_selection = i
        self.cbo_language.SetSelection(lang_selection)

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.lbl_language, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        layout.Add(self.cbo_language, flag=wx.ALL, border=2)
        layout.AddSpacer(8)
        self.pnl_preset.SetSizer(layout)

        # main layout

        layout = wx.GridBagSizer(vgap=0, hgap=0)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.pnl_preset, (0, 0), flag=wx.ALIGN_RIGHT | wx.TOP, border=0)
        layout.Add(self.pnl_options, (1, 0), flag=wx.ALL | wx.EXPAND, border=8)
        layout.Add(self.pnl_process, (2, 0), flag=wx.ALL | wx.EXPAND, border=8)
        self.SetSizer(layout)

        # bind
        self._bind_desktop_events()
        self.cbo_language.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_language)

        self.chk_bind_addr.Bind(wx.EVT_CHECKBOX, self.update_bind_addr_state)
        self.txt_bind_addr.Bind(wx.EVT_TEXT, self.update_bind_addr_warning)
        self.lbl_bind_addr_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_bind_addr_warning)
        self.chk_auth.Bind(wx.EVT_CHECKBOX, self.update_auth_state)

        self.tgl_adj_fine.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle_finetuning)
        self.btn_adj_align.Bind(wx.EVT_BUTTON, self.on_click_align)
        self.sld_adj_divergence.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)
        self.sld_adj_convergence.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)
        self.sld_adj_edge_dilation.Bind(wx.EVT_SPINCTRL, self.update_args_adjustment)
        self.cbo_adj_edge_dilation_y.Bind(wx.EVT_TEXT, self.update_args_adjustment)
        self.sld_adj_foreground_scale.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)

        self.cbo_view_mode.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_view_mode)

        self.btn_detect_ip.Bind(wx.EVT_BUTTON, self.on_click_btn_detect_ip)
        self.btn_url.Bind(wx.EVT_BUTTON, self.on_click_btn_url)
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_FPS, self.on_fps)
        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_options.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()
        self.load_preset()
        self.grp_adjustment.Hide()

        self.update_bind_addr_state()
        self.update_bind_addr_warning()
        self.update_depth_model_list()
        self.update_stereo_format()
        self.update_auth_state()

        self.update_edge_dilation()
        self.update_ema_normalize()
        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_view_mode()
        self.update_monitor_index()
        self.update_window_names()
        self.update_finetuning()
        self.update_compile()

        self.update_wayland_warning()

        self.btn_start.SetFocus()

    def get_editable_comboboxes(self):
        return self.get_desktop_editable_comboboxes()

    def on_close(self, event):
        self.save_preset()
        event.Skip()

        self.stop_event.set()
        if self.processing:
            max_wait = int(6 / 0.01)
            for _ in range(max_wait):
                if not self.stop_event.is_set():
                    break
                time.sleep(0.01)
            if self.stop_event.is_set():
                # It may be deadlocked, so force exit
                os._exit(-1)

    def sync_adj_controls(self, to=True):
        if to:
            divergence = float(self.cbo_divergence.GetValue())
            convergence = float(self.cbo_convergence.GetValue())
            foreground_scale = float(self.cbo_foreground_scale.GetValue())
            edge_dilation = int(self.cbo_edge_dilation.GetValue())
            edge_dilation_y = self.cbo_edge_dilation_y.GetValue()

            self.sld_adj_divergence.SetValue(divergence)
            self.sld_adj_convergence.SetValue(convergence)
            self.sld_adj_foreground_scale.SetValue(foreground_scale)
            self.sld_adj_edge_dilation.SetValue(edge_dilation)
            self.cbo_adj_edge_dilation_y.SetValue(edge_dilation_y)
            self._last_synced_adj = read_runtime_adjustment_values(self.args, self.args_lock)
        else:
            divergence = self.sld_adj_divergence.GetValue()
            convergence = self.sld_adj_convergence.GetValue()
            foreground_scale = self.sld_adj_foreground_scale.GetValue()
            edge_dilation = self.sld_adj_edge_dilation.GetValue()
            edge_dilation_y = self.cbo_adj_edge_dilation_y.GetValue()
            self.cbo_divergence.SetValue(str(round(float(divergence), 4)))
            self.cbo_convergence.SetValue(str(round(float(convergence), 4)))
            self.cbo_foreground_scale.SetValue(str(round(float(foreground_scale), 4)))
            self.cbo_edge_dilation.SetValue(str(edge_dilation))
            self.cbo_edge_dilation_y.SetValue(edge_dilation_y)

    def _apply_args_to_adj_controls(self, args):
        current = read_runtime_adjustment_values(args, self.args_lock)
        if current is None or current == self._last_synced_adj:
            return
        divergence, convergence, foreground_scale, edge_x, edge_y = current
        prev = self._last_synced_adj
        if prev is None or prev[0] != divergence:
            self.sld_adj_divergence.SetValue(divergence)
        if prev is None or prev[1] != convergence:
            self.sld_adj_convergence.SetValue(convergence)
        if prev is None or prev[2] != foreground_scale:
            self.sld_adj_foreground_scale.SetValue(foreground_scale)
        if prev is None or prev[3] != edge_x:
            self.sld_adj_edge_dilation.SetValue(edge_x)
        if prev is None or prev[4] != edge_y:
            self.cbo_adj_edge_dilation_y.SetValue(edge_y)
        self._last_synced_adj = current

    def update_args_adjustment(self, event):
        if self.args is None:
            return
        with self.args_lock:
            divergence = self.sld_adj_divergence.GetValue()
            convergence = self.sld_adj_convergence.GetValue()
            foreground_scale = self.sld_adj_foreground_scale.GetValue()
            edge_dilation = self.sld_adj_edge_dilation.GetValue()
            edge_dilation_y = self.cbo_adj_edge_dilation_y.GetValue()
            # update
            self.args.divergence = divergence
            self.args.convergence = convergence
            self.args.foreground_scale = foreground_scale

            if edge_dilation_y:
                try:
                    edge_y_int = int(edge_dilation_y)
                    self.args.edge_dilation = [edge_dilation, edge_y_int]
                    edge_y_str = str(edge_y_int)
                except ValueError:
                    self.args.edge_dilation = edge_dilation
                    edge_y_str = ""
            else:
                self.args.edge_dilation = edge_dilation
                edge_y_str = ""

            if self.args.state["depth_model"]:
                is_metric = self.args.state["depth_model"].is_metric()
                self.args.mapper = IW3U.resolve_mapper_name(
                    mapper=None, foreground_scale=self.args.foreground_scale, metric_depth=is_metric
                )
            self._last_synced_adj = (
                float(divergence),
                float(convergence),
                float(foreground_scale),
                int(edge_dilation),
                edge_y_str,
            )

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

    def on_click_bind_addr_warning(self, event):
        self.lbl_bind_addr_warning.Hide()
        self.GetSizer().Layout()

    def update_auth_state(self, *args, **kwargs):
        if self.chk_auth.IsChecked():
            self.txt_auth_username.Enable()
            self.txt_auth_password.Enable()
        else:
            self.txt_auth_username.Disable()
            self.txt_auth_password.Disable()

    def on_click_btn_detect_ip(self, event):
        self.txt_bind_addr.SetValue(get_local_address())

    def on_click_btn_url(self, event):
        url = self.txt_url.GetValue()
        if url:
            start_file(url)

    def parse_args(self):
        if not self.validate_desktop_stream_inputs(validate_port=True):
            return None

        parser = create_parser()
        bind_addr = self.txt_bind_addr.GetAddress() if self.chk_bind_addr.IsChecked() else None
        if self.chk_auth.IsChecked():
            user = self.txt_auth_username.GetValue()
            password = self.txt_auth_password.GetValue()
        else:
            user = password = None

        defaults = self.collect_desktop_stream_defaults()
        view_mode = self.cbo_view_mode.GetValue()
        local_viewer = view_mode == "Local Viewer"
        legacy_stream = view_mode == "Web Streaming (Legacy)"
        if local_viewer:
            viewer_kwargs = dict(
                local_viewer=True,
                uncap_fps=self.chk_uncap_fps.GetValue(),
                fit_window=self.chk_fit_window.GetValue(),
            )
        else:
            viewer_kwargs = dict(
                legacy_stream=legacy_stream,
                bind_addr=bind_addr,
                port=self.txt_port.GetValue(),
                user=user,
                password=password,
            )

        parser.set_defaults(**defaults, **viewer_kwargs)
        args = parser.parse_args()
        set_state_args(
            args,
            args_lock=self.args_lock,
            fps_event=FPSGUI(self),
            stop_event=self.stop_event,
            tqdm_fn=functools.partial(TQDMGUI, self),
            depth_model=self.depth_model,
        )
        return args

    def on_fps(self, event):
        estimated_fps, screenshot_fps, streaming_fps, screen_size, url = event.GetValue()
        if estimated_fps is not None and screenshot_fps is not None and streaming_fps is not None:
            self.SetStatusText(
                f"Estimated FPS: {estimated_fps:.02f},"
                f" Screenshot FPS: {screenshot_fps:.02f},"
                f" Streaming FPS: {streaming_fps:.02f},"
                f" Screen: {screen_size}"
            )
        if url:
            self.txt_url.SetValue(url)
        if self.args is not None:
            self._apply_args_to_adj_controls(self.args)

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            self._tqdm_total = value if 0 < value else 1
            self._tqdm_pos = 0
            unit = getattr(event, "unit", "FPS")
            self.fps_tracker.reset(start_pos=0, unit=unit)
            if unit == "KiB":
                self.SetStatusText(f"0.0/{value / 1024.0:.1f} MiB {desc}")
            else:
                self.SetStatusText(f"{0}/{value} {desc}")
        elif type == 1:
            self._tqdm_pos += value
            if self._tqdm_pos > self._tqdm_total:
                self._tqdm_total = self._tqdm_pos
            status_text = self.fps_tracker.update(self._tqdm_pos, self._tqdm_total, desc)
            if status_text is not None:
                self.SetStatusText(status_text)
        elif type == 2:
            if not self.stop_event.is_set() and self.args is not None:
                self.SetStatusText(f"Loading {self.args.depth_model}...")

    def on_click_btn_start(self, event):
        self.args = self.parse_args()
        if self.args is None:
            return

        self.stop_event.clear()
        self.btn_cancel.Enable()
        self.btn_cancel.SetFocus()
        self.btn_start.Disable()
        self.btn_url.Enable()
        self.pnl_preset.Hide()
        self.grp_stereo.Hide()
        self.grp_processor.Hide()
        self.grp_network.Hide()
        self.grp_adjustment.Show()
        self.sync_adj_controls(to=True)
        refresh_layouts(self)
        wx.CallAfter(refresh_layouts, self)

        if self._BaseDepthModel.has_model_checkpoint_file(self.args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {self.args.depth_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {self.args.depth_model}...")

        startWorker(self.on_exit_worker, iw3_desktop_main, wargs=(self.args, False))
        self.processing = True

    def on_exit_worker(self, result):
        error_info = None
        final_args = None
        try:
            args = result.get()
            final_args = args
            self.sync_depth_model_from_args(args)
            self.SetStatusText(T("Shutdown"))
        except:  # noqa
            final_args = self.args
            self.sync_depth_model_from_args(self.args)
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            print(e, file=sys.stderr)
            traceback.print_tb(tb)
            error_info = (message, f"{T('Error')}: {e.__class__.__name__}")

        if final_args is not None:
            self._apply_args_to_adj_controls(final_args)
        self.args = None
        self.processing = False
        self.btn_start.Enable()
        self.btn_start.SetFocus()
        self.btn_cancel.Disable()
        self.txt_url.SetValue("")
        self.btn_url.Disable()
        self.pnl_preset.Show()
        self.grp_stereo.Show()
        self.grp_processor.Show()
        self.grp_network.Show()
        self.grp_adjustment.Hide()
        self.sync_adj_controls(to=False)
        self._last_synced_adj = None
        refresh_layouts(self)
        wx.CallAfter(refresh_layouts, self)

        if error_info is not None:
            wx.MessageBox(error_info[0], error_info[1], wx.OK | wx.ICON_ERROR)
            refresh_layouts(self)
            wx.CallAfter(refresh_layouts, self)

        # free vram
        gc_collect()

    def on_click_btn_cancel(self, event):
        self.stop_event.set()

    def save_preset(self, name=None):
        if not name:
            name = ""
            config_file = CONFIG_PATH
        else:
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")
            if path.exists(config_file):
                with wx.MessageDialog(
                    None, message=name + "\n" + T("already exists. Overwrite?"), caption=T("Confirm"), style=wx.YES_NO
                ) as dlg:
                    if dlg.ShowModal() != wx.ID_YES:
                        return

        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(config_file)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        manager.SaveAndUnregister()

    def load_preset(self, name=None, exclude_names=set()):
        exclude_names.add("cbo_language")  # ignore language
        if not name:
            name = ""
            config_file = CONFIG_PATH
        else:
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")

        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(config_file)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        persistent_manager_restore_all(manager, exclude_names)
        persistent_manager_unregister_all(manager)
        self.ensure_valid_video_codec_selection()

    def on_text_changed_cbo_language(self, event):
        lang = self.cbo_language.GetClientData(self.cbo_language.GetSelection())
        save_language_setting(LANG_CONFIG_PATH, lang)
        with wx.MessageDialog(
            None, message=T("The language setting will be applied after restarting"), style=wx.OK
        ) as dlg:
            dlg.ShowModal()

    def on_text_changed_cbo_view_mode(self, event):
        self.update_view_mode()

    def update_view_mode(self, *args, **kwargs):
        view_mode = self.cbo_view_mode.GetValue()
        local_viewer = view_mode == "Local Viewer"
        legacy_stream = view_mode == "Web Streaming (Legacy)"
        target_codecs = (
            list(self._available_legacy_video_codecs) if legacy_stream else list(self._available_stream_video_codecs)
        )
        current_codecs = self.cbo_video_codec.GetItems()
        if current_codecs != target_codecs:
            current_codec = self.cbo_video_codec.GetValue()
            if current_codec:
                if current_codecs == self._available_legacy_video_codecs:
                    self._last_legacy_video_codec = current_codec
                else:
                    self._last_webxr_video_codec = current_codec
            preferred = self._last_legacy_video_codec if legacy_stream else self._last_webxr_video_codec
            if preferred not in target_codecs:
                preferred = current_codec if current_codec in target_codecs else None
            self.cbo_video_codec.SetItems(target_codecs)
            if preferred in target_codecs:
                self.cbo_video_codec.SetSelection(target_codecs.index(preferred))
            elif target_codecs:
                self.cbo_video_codec.SetSelection(0)

        streaming_options = [
            self.chk_bind_addr,
            self.txt_bind_addr,
            self.btn_detect_ip,
            self.lbl_port,
            self.txt_port,
            self.lbl_stream_quality,
            self.cbo_stream_quality,
            self.lbl_video_codec,
            self.cbo_video_codec,
            self.sep_network1,
            self.chk_auth,
            self.lbl_auth_username,
            self.txt_auth_username,
            self.lbl_auth_password,
            self.txt_auth_password,
        ]
        local_viewer_options = [
            self.chk_uncap_fps,
            self.chk_fit_window,
        ]

        if local_viewer:
            for control in streaming_options:
                control.Hide()
            for control in local_viewer_options:
                control.Show()
        else:
            for control in streaming_options:
                control.Show()
            for control in local_viewer_options:
                control.Hide()

        if self.IsShown():
            refresh_layouts(self)

    def update_finetuning(self):
        # Fine-tuning spin controls
        if self.tgl_adj_fine.GetValue():
            self.sld_adj_divergence.SetIncrement(DIVERGENCE_INC_FINE)
            self.sld_adj_convergence.SetIncrement(CONVERGENCE_INC_FINE)
            self.sld_adj_foreground_scale.SetIncrement(FOREGROUNDSCALE_INC_FINE)
        else:
            self.sld_adj_divergence.SetIncrement(DIVERGENCE_INC_DEFAULT)
            self.sld_adj_convergence.SetIncrement(CONVERGENCE_INC_DEFAULT)
            self.sld_adj_foreground_scale.SetIncrement(FOREGROUNDSCALE_INC_DEFAULT)

    def on_toggle_finetuning(self, event):
        # Fine-tuning spin controls
        self.update_finetuning()

    def on_click_align(self, event):
        def align(w):
            inc = w.GetIncrement()
            value = w.GetValue()
            w.SetValue(math.floor(round(value / inc, 10)) * inc)

        align(self.sld_adj_divergence)
        align(self.sld_adj_convergence)
        align(self.sld_adj_foreground_scale)

    @staticmethod
    def set_flat_size(w):
        size = w.GetSize()
        new_size = (size.width, int(size.height * 0.8))
        w.SetMinSize(new_size)
        w.SetSize(new_size)

        font = w.GetFont()
        font.SetPointSize(int(font.GetPointSize() * 0.8))
        w.SetFont(font)


LOCALE_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse
    import sys

    global LOCALE_DICT

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, choices=list(LOCALES.keys()), help="translation language")
    args = parser.parse_args()
    if args.lang:
        LOCALE_DICT = LOCALES.get(args.lang, {})
    else:
        saved_lang = load_language_setting(LANG_CONFIG_PATH)
        if saved_lang:
            LOCALE_DICT = LOCALES.get(saved_lang, {})

    sys.argv = [sys.argv[0]]  # clear command arguments

    app = IW3DesktopApp()
    app.MainLoop()


if __name__ == "__main__":
    init_win32()
    main()
