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

from ..desktop.gui_shared import EVT_FPS, FPSGUI, DesktopStreamGUIMixin
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


class MainFrame(wx.Frame, DesktopStreamGUIMixin):
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
        self._last_synced_stream_settings = None
        self.network_stream_widgets: list[wx.Window] = []
        self.processing = False
        self.args = None
        self.args_lock = threading.Lock()
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

    def _T(self, s: str) -> str:
        return T(s)

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
        for i, lang in enumerate(LOCALE_LIST):
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

        self.load_settings()
        self.update_bind_addr_state()
        self.update_bind_addr_warning()
        self.update_auth_state()
        self.update_stream_enable_state()

        self.btn_start.SetFocus()

    def _init_desktop_ui(self):
        if self._desktop_ui_initialized:
            return

        self.network_stream_widgets = self._build_stream_network_controls(
            self.grp_network, self.network_grid, start_row=7
        )
        self.sizer_stereo = self._build_stereo_group(self.pnl_options, self._warning_font, self._warning_color)
        self.sizer_processor = self._build_processor_group(self.pnl_options)

        self.options_layout.Add(self.sizer_stereo, (0, 0), (2, 1), flag=wx.ALL | wx.EXPAND, border=4)
        self.options_layout.Add(self.sizer_processor, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)

        self._bind_desktop_events()

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

    def get_editable_comboboxes(self):
        if not self._desktop_ui_initialized:
            return []
        return self.get_desktop_editable_comboboxes()

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

    def on_text_changed_cbo_language(self, event):
        lang = self.cbo_language.GetClientData(self.cbo_language.GetSelection())
        save_language_setting(LANG_CONFIG_PATH, lang)
        with wx.MessageDialog(
            None, message=T("The language setting will be applied after restarting"), style=wx.OK
        ) as dlg:
            dlg.ShowModal()

    def on_open_url(self, event):
        url = self.txt_url.GetValue()
        if url:
            start_file(url)

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
        base_defaults = dict(
            root=root,
            bind_addr=bind_addr,
            port=int(self.txt_port.GetValue()),
            user=user,
            password=password,
            enable_stream=enable_stream,
        )

        if not enable_stream:
            parser = create_parser(enable_stream=False)
            parser.set_defaults(**base_defaults)
            args = parser.parse_args([])
            set_state_args(args, args_lock=self.args_lock, fps_event=FPSGUI(self), stop_event=self.stop_event)
            return args

        self._init_desktop_ui()
        if not self.validate_desktop_stream_inputs(validate_port=False):
            return None

        parser = create_parser(enable_stream=True)
        parser.set_defaults(**base_defaults, **self.collect_desktop_stream_defaults())
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
        if not self._desktop_ui_initialized or args is None or not getattr(args, "enable_stream", False):
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
            self.SetStatusText(T("Running"))

    def on_start(self, event):
        self.args = self.parse_args()
        if self.args is None:
            return

        if self._desktop_ui_initialized and getattr(self.args, "enable_stream", False):
            self._last_synced_stream_settings = self._read_runtime_adjustment_values(self.args)
        else:
            self._last_synced_stream_settings = None

        self.stop_event.clear()
        self.btn_stop.Enable()
        self.btn_stop.SetFocus()
        self.btn_start.Disable()
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
            if self._desktop_ui_initialized:
                self.sync_depth_model_from_args(args)
            self.SetStatusText(T("Shutdown"))
        except Exception:
            final_args = self.args
            if self._desktop_ui_initialized:
                self.sync_depth_model_from_args(self.args)
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
        self.btn_start.SetFocus()
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


LOCALE_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


if __name__ == "__main__":
    init_win32_dpi()
    main()
