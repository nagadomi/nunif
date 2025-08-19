import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import locale
import sys
import os
import time
from os import path
import traceback
import threading
import importlib.util
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as persist
import wx.lib.stattext as stattext
from wx.lib.buttons import GenBitmapButton
from wx.lib.intctrl import IntCtrl
import torch
from nunif.utils.git import get_current_branch
from nunif.initializer import gc_collect
from nunif.device import mps_is_available, xpu_is_available, create_device
from nunif.models.utils import check_compile_support
from nunif.utils.filename import sanitize_filename
from nunif.gui import (
    IpAddrCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_unregister_all,
    persistent_manager_restore_all, persistent_manager_register,
    validate_number,
    set_icon_ex, load_icon, start_file,
    apply_dark_mode, is_dark_mode,
)
from ..depth_anything_model import DepthAnythingModel
from ..video_depth_anything_streaming_model import VideoDepthAnythingStreamingModel
from ..locales import LOCALES, load_language_setting, save_language_setting
from .utils import (
    get_local_address,
    is_private_address,
    is_loopback_address,
    init_win32,
    iw3_desktop_main,
    create_parser, set_state_args,
    get_monitor_size_list,
    enum_window_names,
    IW3U, ENABLE_GPU_JPEG,
)


CONFIG_DIR = path.join(path.dirname(__file__), "..", "..", "tmp")
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-desktop.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-desktop-lang.cfg")
PRESET_DIR = path.join(CONFIG_DIR, "presets")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

LAYOUT_DEBUG = False
HAS_WINDOWS_CAPTURE = importlib.util.find_spec("windows_capture")


myEVT_FPS = wx.NewEventType()
EVT_FPS = wx.PyEventBinder(myEVT_FPS, 0)


class FPSEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, estimated_fps=None, screenshot_fps=None, streaming_fps=None,
                 screen_size=None, url=None):
        super(FPSEvent, self).__init__(etype, eid)
        self.estimated_fps = estimated_fps
        self.screenshot_fps = screenshot_fps
        self.streaming_fps = streaming_fps
        self.screen_size = screen_size
        self.url = url

    def GetValue(self):
        return (self.estimated_fps, self.screenshot_fps, self.streaming_fps, self.screen_size, self.url)


class FPSGUI():
    def __init__(self, parent, **kwargs):
        self.parent = parent

    def set_url(self, url):
        wx.PostEvent(self.parent, FPSEvent(myEVT_FPS, -1, None, None, None, None, url))

    def update(self, estimated_fps, screenshot_fps, streaming_fps, screen_size):
        wx.PostEvent(self.parent, FPSEvent(myEVT_FPS, -1, estimated_fps, screenshot_fps, streaming_fps,
                                           screen_size, None))


class IW3DesktopApp(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker(main_frame.GetTitle())
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(None,
                                  message=T("Another instance is running"),
                                  caption=T("Confirm"), style=wx.OK) as dlg:
                dlg.ShowModal()
                return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
        self.SetTopWindow(main_frame)

        return True


class MainFrame(wx.Frame):
    def __init__(self):
        branch_name = get_current_branch()
        if branch_name is None or branch_name in {"master", "main"}:
            branch_tag = ""
        else:
            branch_tag = f" ({branch_name})"

        super(MainFrame, self).__init__(
            None,
            name="iw3-desktop",
            title=T("iw3-desktop") + branch_tag,
            size=(720, 560),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.stop_event = threading.Event()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.processing = False
        self.args = None
        self.args_lock = threading.Lock()
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

    def initialize_component(self):
        NORMAL_FONT = wx.Font(10, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_FONT = wx.Font(8, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_COLOR = (0xcc, 0x33, 0x33)

        self.SetFont(NORMAL_FONT)
        self.CreateStatusBar()

        # options panel
        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # stereo generation settings

        self.grp_stereo = wx.StaticBox(self.pnl_options, label=T("Stereo Generation"))
        if LAYOUT_DEBUG:
            self.grp_stereo.SetBackgroundColour("#fcf")

        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = EditableComboBox(self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"],
                                               name="cbo_divergence")
        self.lbl_divergence_warning = stattext.GenStaticText(self.grp_stereo, label="")
        self.lbl_divergence_warning.SetFont(WARNING_FONT)
        self.lbl_divergence_warning.SetForegroundColour(WARNING_COLOR)
        self.lbl_divergence_warning.Hide()

        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(5)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence = EditableComboBox(self.grp_stereo, choices=["0.0", "0.5", "1.0"],
                                                name="cbo_convergence")
        self.cbo_convergence.SetSelection(2)
        self.cbo_convergence.SetToolTip("Convergence")

        self.lbl_synthetic_view = wx.StaticText(self.grp_stereo, label=T("Synthetic View"))
        self.cbo_synthetic_view = wx.ComboBox(self.grp_stereo,
                                              choices=["both", "right", "left"],
                                              name="cbo_synthetic_view")
        self.cbo_synthetic_view.SetEditable(False)
        self.cbo_synthetic_view.SetSelection(0)

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo,
                                      choices=["mlbw_l2", "mlbw_l4", "mlbw_l2s",
                                               "row_flow_v3", "row_flow_v3_sym", "row_flow_v2",
                                               "forward_fill"],
                                      name="cbo_method")
        self.cbo_method.SetEditable(False)
        self.cbo_method.SetSelection(4)

        self.chk_small_model_only = wx.CheckBox(self.grp_stereo, label=T("List small model only"), name="chk_small_model_only")
        self.chk_small_model_only.SetValue(True)
        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))

        depth_models = self.get_depth_models(small_only=False)
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
                                           choices=depth_models,
                                           name="cbo_depth_model")
        self.cbo_depth_model.SetEditable(False)
        self.cbo_depth_model.SetSelection(depth_models.index("Any_V2_S"))

        self.lbl_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_resolution = EditableComboBox(self.grp_stereo,
                                               choices=["Default", "512"],
                                               name="cbo_zoed_resolution")
        self.cbo_resolution.SetSelection(0)

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(self.grp_stereo,
                                                     choices=["-3", "-2", "-1", "0", "1", "2", "3"],
                                                     name="cbo_foreground_scale")
        self.cbo_foreground_scale.SetSelection(3)

        self.chk_edge_dilation = wx.CheckBox(self.grp_stereo, label=T("Edge Fix"), name="chk_edge_dilation")
        self.cbo_edge_dilation = EditableComboBox(self.grp_stereo,
                                                  choices=["0", "1", "2", "3", "4"],
                                                  name="cbo_edge_dilation")
        self.chk_edge_dilation.SetValue(True)

        self.cbo_edge_dilation.SetSelection(2)
        self.cbo_edge_dilation.SetToolTip(T("Reduce distortion of foreground and background edges"))

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
                                             label=T("Flicker Reduction"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetValue(True)
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))
        self.cbo_ema_decay = EditableComboBox(self.grp_stereo, choices=["0.99", "0.95", "0.9", "0.75", "0.5"],
                                              name="cbo_ema_decay")
        self.cbo_ema_decay.SetSelection(2)
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))

        self.chk_preserve_screen_border = wx.CheckBox(self.grp_stereo,
                                                      label=T("Preserve Screen Border"),
                                                      name="chk_preserve_screen_border")
        self.chk_preserve_screen_border.SetValue(True)
        self.chk_preserve_screen_border.SetToolTip(T("Force set screen border parallax to zero"))

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(
            self.grp_stereo,
            choices=["Half SBS", "Full SBS", "RGB-D", "Half RGB-D"],
            name="cbo_stereo_format")
        self.cbo_stereo_format.SetEditable(False)
        self.cbo_stereo_format.SetSelection(0)
        self.lbl_format_device = wx.StaticText(self.grp_stereo, label=T(""))

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))

        i = 0
        layout.Add(self.lbl_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_divergence_warning, pos=(i := i + 1, 0), span=(0, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_convergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_synthetic_view, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_synthetic_view, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_small_model_only, (i := i + 1, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_depth_model, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_model, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_resolution, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_resolution, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_foreground_scale, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_scale, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_edge_dilation, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_edge_dilation, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_ema_normalize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_ema_decay, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_preserve_screen_border, (i := i + 1, 0), (0, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_stereo_format, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_format_device, (i := i + 1, 1), flag=wx.ALIGN_CENTER_VERTICAL)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # network settings
        self.grp_network = wx.StaticBox(self.pnl_options, label=T("Network"))
        if LAYOUT_DEBUG:
            self.grp_network.SetBackgroundColour("#fcf")

        self.lbl_view_mode = wx.StaticText(self.grp_network, label=T("View Mode"))
        self.cbo_view_mode = wx.ComboBox(self.grp_network, choices=["Web Streaming", "Local Viewer"],
                                         name="cbo_view_mode")
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
        self.txt_port = IntCtrl(self.grp_network, size=self.FromDIP((200, -1)),
                                allow_none=False, min=1025, max=65535, name="txt_port")
        self.txt_port.SetValue(1303)
        self.lbl_stream_fps = wx.StaticText(self.grp_network, label=T("Streaming FPS"))
        self.cbo_stream_fps = EditableComboBox(self.grp_network, choices=["30", "24", "15", "8"],
                                               name="cbo_stream_fps")
        self.cbo_stream_fps.SetSelection(0)

        self.lbl_stream_height = wx.StaticText(self.grp_network, label=T("Streaming Resolution"))
        self.cbo_stream_height = EditableComboBox(self.grp_network, choices=["1080", "720"],
                                                  name="cbo_stream_height")
        self.cbo_stream_height.SetSelection(0)

        self.lbl_stream_quality = wx.StaticText(self.grp_network, label=T("MJPEG Quality"))
        self.cbo_stream_quality = EditableComboBox(self.grp_network, choices=["100", "95", "90", "85", "80"],
                                                   name="cbo_stream_quality")
        self.cbo_stream_quality.SetSelection(2)
        self.chk_pad_16_9 = wx.CheckBox(self.grp_network, label=T("Padding") + " 16:9", name="chk_pad_16_9")
        self.chk_pad_16_9.SetValue(False)

        self.chk_gpu_jpeg = wx.CheckBox(self.grp_network, label=T("GPU JPEG"), name="chk_gpu_jpeg")
        self.chk_gpu_jpeg.SetValue(False)
        if not ENABLE_GPU_JPEG:
            self.chk_gpu_jpeg.Disable()
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
        layout.Add(self.lbl_stream_height, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stream_height, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_stream_quality, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stream_quality, (6, 1), flag=wx.EXPAND)
        layout.Add(self.chk_gpu_jpeg, (7, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_pad_16_9, (8, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sep_network1, (9, 0), (0, 3), flag=wx.EXPAND | wx.ALL)

        layout.Add(self.chk_auth, (10, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_auth_username, (11, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_auth_username, (11, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_auth_password, (12, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_auth_password, (12, 1), flag=wx.EXPAND)

        sizer_network = wx.StaticBoxSizer(self.grp_network, wx.VERTICAL)
        sizer_network.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        if LAYOUT_DEBUG:
            self.grp_processor.SetBackgroundColour("#fcf")

        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=self.FromDIP((200, -1)), name="cbo_device")
        self.cbo_device.SetEditable(False)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_device.Append(f"{i}:{device_name}", i)
        elif mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif xpu_is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                self.cbo_device.Append(f"{i}:{device_name}", i)

        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_screenshot = wx.StaticText(self.grp_processor, label=T("Screenshot"))
        screenshot_backends = ["pil", "pil_mp"] + (["wc_mp"] if HAS_WINDOWS_CAPTURE else [])
        self.cbo_screenshot = wx.ComboBox(self.grp_processor,
                                          choices=screenshot_backends,
                                          name="cbo_screenshot")
        self.cbo_screenshot.SetEditable(False)
        if sys.platform == "win32" and HAS_WINDOWS_CAPTURE:
            self.cbo_screenshot.SetSelection(2)
        else:
            self.cbo_screenshot.SetSelection(0)

        self.lbl_monitor_index = wx.StaticText(self.grp_processor, label=T("Monitor Index"))
        self.cbo_monitor_index = wx.ComboBox(self.grp_processor,
                                             choices=[str(i) for i in range(len(get_monitor_size_list()))],
                                             name="cbo_monitor_index")
        self.cbo_monitor_index.SetEditable(False)
        self.cbo_monitor_index.SetSelection(0)

        self.lbl_window_name = wx.StaticText(self.grp_processor, label=T("Window Name"))
        self.cbo_window_name = EditableComboBox(self.grp_processor, size=self.FromDIP((200, -1)), choices=[""],
                                                name="cbo_window_name")
        self.cbo_window_name.SetSelection(0)
        self.btn_reload_window_name = GenBitmapButton(self.grp_processor, bitmap=load_icon("view-refresh.png"))
        self.btn_reload_window_name.SetToolTip(T("Reload"))

        self.lbl_crop_region = wx.StaticText(self.grp_processor, label=T("Crop Region (px)"))

        self.lbl_crop_top = wx.StaticText(self.grp_processor, label=T("Top"))
        self.txt_crop_top = IntCtrl(self.grp_processor, size=self.FromDIP((60, -1)),
                                    allow_none=False, min=0, max=500, name="txt_crop_top")
        self.txt_crop_top.SetValue(0)

        self.lbl_crop_left = wx.StaticText(self.grp_processor, label=T("Left"))
        self.txt_crop_left = IntCtrl(self.grp_processor, size=self.FromDIP((60, -1)),
                                     allow_none=False, min=0, max=500, name="txt_crop_left")
        self.txt_crop_left.SetValue(0)

        self.lbl_crop_right = wx.StaticText(self.grp_processor, label=T("Right"))
        self.txt_crop_right = IntCtrl(self.grp_processor, size=self.FromDIP((60, -1)),
                                      allow_none=False, min=0, max=500, name="txt_crop_right")
        self.txt_crop_right.SetValue(0)

        self.lbl_crop_bottom = wx.StaticText(self.grp_processor, label=T("Bottom"))
        self.txt_crop_bottom = IntCtrl(self.grp_processor, size=self.FromDIP((60, -1)),
                                       allow_none=False, min=0, max=500, name="txt_crop_bottom")
        self.txt_crop_bottom.SetValue(0)

        self.chk_compile = wx.CheckBox(self.grp_processor, label=T("torch.compile"), name="chk_compile")
        self.chk_compile.SetToolTip(T("Enable model compiling"))
        self.chk_compile.SetValue(False)

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_screenshot, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_screenshot, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_monitor_index, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_monitor_index, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_window_name, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_window_name, (3, 1), flag=wx.EXPAND)
        layout.Add(self.btn_reload_window_name, (3, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_crop_region, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        # Create a sub-layout for crop controls
        crop_layout = wx.GridBagSizer(vgap=2, hgap=4)
        crop_layout.Add(self.lbl_crop_top, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_top, (0, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_left, (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_left, (0, 3), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_right, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_right, (1, 1), flag=wx.EXPAND)
        crop_layout.Add(self.lbl_crop_bottom, (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        crop_layout.Add(self.txt_crop_bottom, (1, 3), flag=wx.EXPAND)

        layout.Add(crop_layout, (4, 1), flag=wx.EXPAND)
        layout.Add(self.chk_compile, (5, 0), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # adjustment settings
        self.grp_adjustment = wx.StaticBox(self.pnl_options, label=T("Adjustment"))
        if LAYOUT_DEBUG:
            self.grp_adjustment.SetBackgroundColour("#ccc")

        self.lbl_adj_divergence = wx.StaticText(self.grp_adjustment, label=T("3D Strength"))
        self.sld_adj_divergence = wx.SpinCtrlDouble(self.grp_adjustment, value="1.00", min=0.0, max=5.0, inc=0.25)
        self.sld_adj_divergence.SetDigits(2)
        self.lbl_adj_convergence = wx.StaticText(self.grp_adjustment, label=T("Convergence Plane"))
        self.sld_adj_convergence = wx.SpinCtrlDouble(self.grp_adjustment, value="1.0", min=0.0, max=1.0, inc=0.1)
        self.lbl_adj_foreground_scale = wx.StaticText(self.grp_adjustment, label=T("Foreground Scale"))
        self.sld_adj_foreground_scale = wx.SpinCtrlDouble(self.grp_adjustment, value="0", min=-3.0, max=3.0, inc=0.2)
        self.lbl_adj_edge_dilation = wx.StaticText(self.grp_adjustment, label=T("Edge Fix"))
        self.sld_adj_edge_dilation = wx.SpinCtrl(self.grp_adjustment, value="0", min=0, max=10)

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

        self.txt_url = wx.TextCtrl(self.pnl_process, size=(300, -1), style=wx.TE_READONLY)
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
        for i, lang in enumerate(LOCAL_LIST):
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
        self.cbo_language.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_language)

        self.cbo_divergence.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_synthetic_view.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_method.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_method)
        self.lbl_divergence_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_divergence_warning)

        self.chk_small_model_only.Bind(wx.EVT_CHECKBOX, self.update_depth_model_list)
        self.chk_edge_dilation.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)
        self.chk_bind_addr.Bind(wx.EVT_CHECKBOX, self.update_bind_addr_state)
        self.txt_bind_addr.Bind(wx.EVT_TEXT, self.update_bind_addr_warning)
        self.lbl_bind_addr_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_bind_addr_warning)
        self.chk_auth.Bind(wx.EVT_CHECKBOX, self.update_auth_state)
        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.update_stereo_format)
        self.cbo_screenshot.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_screenshot)
        self.btn_reload_window_name.Bind(wx.EVT_BUTTON, self.update_window_names)

        self.cbo_device.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_device)
        self.chk_compile.Bind(wx.EVT_CHECKBOX, self.update_compile)

        self.sld_adj_divergence.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)
        self.sld_adj_convergence.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)
        self.sld_adj_edge_dilation.Bind(wx.EVT_SPINCTRL, self.update_args_adjustment)
        self.sld_adj_foreground_scale.Bind(wx.EVT_SPINCTRLDOUBLE, self.update_args_adjustment)

        self.cbo_view_mode.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_view_mode)

        self.btn_detect_ip.Bind(wx.EVT_BUTTON, self.on_click_btn_detect_ip)
        self.btn_url.Bind(wx.EVT_BUTTON, self.on_click_btn_url)
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_FPS, self.on_fps)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_options.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()
        self.load_preset()

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
        self.update_compile()

        self.grp_adjustment.Hide()
        self.Fit()

    def get_depth_models(self, small_only):
        if small_only:
            return ["Any_S", "Any_V2_S", "Any_V2_N_S", "Distill_Any_S", "VDA_Stream_S"]
        else:
            depth_models = [
                "ZoeD_N", "ZoeD_K", "ZoeD_NK",
                "ZoeD_Any_N", "ZoeD_Any_K",
                "Any_S", "Any_B", "Any_L",
                "Any_V2_S",
            ]
            if DepthAnythingModel.has_checkpoint_file("Any_V2_B"):
                depth_models.append("Any_V2_B")
            if DepthAnythingModel.has_checkpoint_file("Any_V2_L"):
                depth_models.append("Any_V2_L")

            depth_models += ["Any_V2_N_S", "Any_V2_N_B"]
            if DepthAnythingModel.has_checkpoint_file("Any_V2_N_L"):
                depth_models.append("Any_V2_N_L")
            depth_models += ["Any_V2_K_S", "Any_V2_K_B"]
            if DepthAnythingModel.has_checkpoint_file("Any_V2_K_L"):
                depth_models.append("Any_V2_K_L")

            depth_models += ["Distill_Any_S"]
            if DepthAnythingModel.has_checkpoint_file("Distill_Any_B"):
                depth_models.append("Distill_Any_B")
            if DepthAnythingModel.has_checkpoint_file("Distill_Any_L"):
                depth_models.append("Distill_Any_L")

            depth_models += ["VDA_Stream_S"]
            if VideoDepthAnythingStreamingModel.has_checkpoint_file("VDA_Stream_L"):
                depth_models.append("VDA_Stream_L")

            return depth_models

    def get_editable_comboboxes(self):
        editable_comboboxes = [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_resolution,
            self.cbo_edge_dilation,
            self.cbo_ema_decay,
            self.cbo_foreground_scale,
            self.cbo_stream_fps,
            self.cbo_stream_height,
        ]
        return editable_comboboxes

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

    def sync_adj_controls(self, to=True):
        if to:
            divergence = float(self.cbo_divergence.GetValue())
            convergence = float(self.cbo_convergence.GetValue())
            foreground_scale = float(self.cbo_foreground_scale.GetValue())
            edge_dilation = int(self.cbo_edge_dilation.GetValue())

            divergence = self.sld_adj_divergence.SetValue(divergence)
            convergence = self.sld_adj_convergence.SetValue(convergence)
            foreground_scale = self.sld_adj_foreground_scale.SetValue(foreground_scale)
            edge_dilation = self.sld_adj_edge_dilation.SetValue(edge_dilation)
        else:
            divergence = self.sld_adj_divergence.GetValue()
            convergence = self.sld_adj_convergence.GetValue()
            foreground_scale = self.sld_adj_foreground_scale.GetValue()
            edge_dilation = self.sld_adj_edge_dilation.GetValue()
            self.cbo_divergence.SetValue(str(divergence))
            self.cbo_convergence.SetValue(str(convergence))
            self.cbo_foreground_scale.SetValue(str(foreground_scale))
            self.cbo_edge_dilation.SetValue(str(edge_dilation))

    def update_args_adjustment(self, event):
        if self.args is None:
            return
        with self.args_lock:
            divergence = self.sld_adj_divergence.GetValue()
            convergence = self.sld_adj_convergence.GetValue()
            foreground_scale = self.sld_adj_foreground_scale.GetValue()
            edge_dilation = self.sld_adj_edge_dilation.GetValue()
            # update
            self.args.divergence = divergence
            self.args.convergence = convergence
            self.args.edge_dilation = edge_dilation
            self.args.foreground_scale = foreground_scale

            if self.args.state["depth_model"]:
                is_metric = self.args.state["depth_model"].is_metric()
                self.args.mapper = IW3U.resolve_mapper_name(
                    mapper=None,
                    foreground_scale=self.args.foreground_scale,
                    metric_depth=is_metric)

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

    def update_stereo_format(self, *args, **kwargs):
        stereo_format = self.cbo_stereo_format.GetValue()
        if stereo_format == "Half SBS":
            self.lbl_format_device.SetLabel("Meta Quest 2/3")
        elif stereo_format == "Full SBS":
            self.lbl_format_device.SetLabel("PICO 4")
        else:
            self.lbl_format_device.SetLabel("")

    def update_edge_dilation(self):
        if self.chk_edge_dilation.IsChecked():
            self.cbo_edge_dilation.Enable()
        else:
            self.cbo_edge_dilation.Disable()

    def on_changed_chk_edge_dilation(self, event):
        self.update_edge_dilation()

    def update_preserve_screen_border(self):
        if self.cbo_method.GetValue() in {"row_flow_v2", "row_flow_v3", "row_flow_v3_sym",
                                          "mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
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

    def on_click_btn_detect_ip(self, event):
        self.txt_bind_addr.SetValue(get_local_address())

    def on_click_btn_url(self, event):
        url = self.txt_url.GetValue()
        if url:
            start_file(url)

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
                None,
                message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
                caption=T("Error"),
                style=wx.OK) as dlg:
            dlg.ShowModal()

    def parse_args(self):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return None
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return None
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return None
        if not validate_number(self.cbo_stream_fps.GetValue(), 1, 60, allow_empty=False):
            self.show_validation_error_message(T("Streaming FPS"), 1, 60)
            return None
        if not validate_number(self.cbo_stream_height.GetValue(), 320, 4320, allow_empty=False):
            self.show_validation_error_message(T("Streaming Resolution"), 320, 4320)
            return None
        if not validate_number(self.cbo_stream_quality.GetValue(), 1, 100, allow_empty=False):
            self.show_validation_error_message(T("MJPEG Quality"), 1, 100)
            return None
        if not validate_number(self.txt_port.GetValue(), 1025, 65535, allow_empty=False):
            self.show_validation_error_message(T("Port"), 1025, 65535)
            return None

        crop_top = self.txt_crop_top.GetValue()
        crop_left = self.txt_crop_left.GetValue()
        crop_right = self.txt_crop_right.GetValue()
        crop_bottom = self.txt_crop_bottom.GetValue()

        for crop_name, crop_value in [("Top", crop_top), ("Left", crop_left), ("Right", crop_right), ("Bottom", crop_bottom)]:
            if not validate_number(str(crop_value), 0, 500, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Crop") + " " + T(crop_name), 0, 500)
                return None

        resolution = self.cbo_resolution.GetValue()
        if resolution == "Default" or resolution == "":
            resolution = None
        else:
            if not validate_number(resolution, 192, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 192, 8190)
                return
            resolution = int(resolution)

        parser = create_parser()
        full_sbs = self.cbo_stereo_format.GetValue() == "Full SBS"
        rgbd = self.cbo_stereo_format.GetValue() == "RGB-D"
        half_rgbd = self.cbo_stereo_format.GetValue() == "Half RGB-D"
        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        device_id = [device_id]

        depth_model_type = self.cbo_depth_model.GetValue()
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id or
                                         self.depth_model_height != resolution)):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            gc_collect()

        edge_dilation = int(self.cbo_edge_dilation.GetValue()) if self.chk_edge_dilation.IsChecked() else 0
        preserve_screen_border = self.chk_preserve_screen_border.IsEnabled() and self.chk_preserve_screen_border.IsChecked()
        bind_addr = self.txt_bind_addr.GetAddress() if self.chk_bind_addr.IsChecked() else None
        if self.chk_auth.IsChecked():
            user = self.txt_auth_username.GetValue()
            password = self.txt_auth_password.GetValue()
        else:
            user = password = None

        monitor_index = int(self.cbo_monitor_index.GetValue())
        if self.cbo_screenshot.GetValue() != "wc_mp":
            monitor_index = 0
        window_name = self.cbo_window_name.GetValue()
        if not window_name:
            window_name = None

        pad_mode = "16:9" if self.chk_pad_16_9.IsChecked() else None
        local_viewer = self.cbo_view_mode.GetValue() == "Local Viewer"
        if local_viewer:
            viewer_kwargs = dict(
                local_viewer=True,
                stream_fps=int(self.cbo_stream_fps.GetValue()),
                stream_height=int(self.cbo_stream_height.GetValue()),
            )
        else:
            viewer_kwargs = dict(
                bind_addr=bind_addr,
                port=self.txt_port.GetValue(),
                user=user,
                password=password,
                stream_fps=int(self.cbo_stream_fps.GetValue()),
                stream_height=int(self.cbo_stream_height.GetValue()),
                stream_quality=int(self.cbo_stream_quality.GetValue()),
                gpu_jpeg=self.chk_gpu_jpeg.IsEnabled() and self.chk_gpu_jpeg.IsChecked(),
            )

        parser.set_defaults(
            gpu=device_id,
            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            synthetic_view=self.cbo_synthetic_view.GetValue(),
            method=self.cbo_method.GetValue(),
            preserve_screen_border=preserve_screen_border,
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            edge_dilation=edge_dilation,
            ema_normalize=self.chk_ema_normalize.GetValue(),
            ema_decay=float(self.cbo_ema_decay.GetValue()),
            resolution=resolution,
            compile=self.chk_compile.IsEnabled() and self.chk_compile.IsChecked(),

            screenshot=self.cbo_screenshot.GetValue(),
            monitor_index=monitor_index,
            window_name=window_name,
            crop_top=crop_top,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
            full_sbs=full_sbs,
            rgbd=rgbd,
            half_rgbd=half_rgbd,
            pad_mode=pad_mode,

            **viewer_kwargs,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            args_lock=self.args_lock,
            fps_event=FPSGUI(self),
            stop_event=self.stop_event,
            depth_model=self.depth_model)
        return args

    def on_fps(self, event):
        estimated_fps, screenshot_fps, streaming_fps, screen_size, url = event.GetValue()
        if estimated_fps is not None and screenshot_fps is not None and streaming_fps is not None:
            self.SetStatusText(f"Estimated FPS: {estimated_fps:.02f},"
                               f" Screenshot FPS: {screenshot_fps:.02f},"
                               f" Streaming FPS: {streaming_fps:.02f},"
                               f" Screen: {screen_size}")
        if url:
            self.txt_url.SetValue(url)

    def on_click_btn_start(self, event):
        self.args = self.parse_args()
        if self.args is None:
            return

        self.stop_event.clear()
        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.btn_url.Enable()
        self.pnl_preset.Hide()
        self.grp_stereo.Hide()
        self.grp_processor.Hide()
        self.grp_network.Hide()
        self.grp_adjustment.Show()
        self.sync_adj_controls(to=True)
        self.Fit()

        if self.args.state["depth_model"].has_checkpoint_file(self.args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {self.args.depth_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {self.args.depth_model}...")

        startWorker(self.on_exit_worker, iw3_desktop_main, wargs=(self.args, False))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            args = result.get()
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.resolution

            self.SetStatusText(T("Shutdown"))
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.args = None
        self.processing = False
        self.btn_cancel.Disable()
        self.btn_start.Enable()
        self.txt_url.SetValue("")
        self.btn_url.Disable()
        self.pnl_preset.Show()
        self.grp_stereo.Show()
        self.grp_processor.Show()
        self.grp_network.Show()
        self.grp_adjustment.Hide()
        self.sync_adj_controls(to=False)
        self.Fit()

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
                with wx.MessageDialog(None,
                                      message=name + "\n" + T("already exists. Overwrite?"),
                                      caption=T("Confirm"), style=wx.YES_NO) as dlg:
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

    def on_click_divergence_warning(self, event):
        self.lbl_divergence_warning.Hide()
        self.GetSizer().Layout()

    def update_divergence_warning(self, *args, **kwargs):
        try:
            divergence = float(self.cbo_divergence.GetValue())
            method = self.cbo_method.GetValue()
            synthetic_view = self.cbo_synthetic_view.GetValue()
            max_divergence = float("inf")

            if method in {"row_flow_v3", "row_flow_v3_sym"}:
                if synthetic_view == "both":
                    max_divergence = 5.0
                else:
                    max_divergence = 5.0 * 0.5
            elif method == "row_flow_v2":
                if synthetic_view == "both":
                    max_divergence = 2.5
                else:
                    max_divergence = 2.5 * 0.5
            elif method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
                if synthetic_view == "both":
                    max_divergence = 10.0
                else:
                    max_divergence = 10.0 * 0.5

            if divergence > max_divergence:
                self.lbl_divergence_warning.SetLabel(
                    f"{divergence}: " + T("Out of range of training data") + f": {method}, {synthetic_view}"
                )
                self.lbl_divergence_warning.SetToolTip(
                    T("This result could be unstable"),
                )
                self.lbl_divergence_warning.Show()
            else:
                self.lbl_divergence_warning.SetLabel("")
                self.lbl_divergence_warning.SetToolTip("")
                self.lbl_divergence_warning.Hide()

            self.GetSizer().Layout()
        except ValueError:
            pass

    def on_selected_index_changed_cbo_device(self, event):
        self.update_compile()

    def update_compile(self, *args, **kwargs):
        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device_id == -2:
            # currently "All CUDA" does not support compile
            self.chk_compile.SetValue(False)
        else:
            # check compiler support
            if self.chk_compile.IsChecked():
                device = create_device(device_id)
                if not check_compile_support(device):
                    self.chk_compile.SetValue(False)

    def on_text_changed_cbo_language(self, event):
        lang = self.cbo_language.GetClientData(self.cbo_language.GetSelection())
        save_language_setting(LANG_CONFIG_PATH, lang)
        with wx.MessageDialog(None,
                              message=T("The language setting will be applied after restarting"),
                              style=wx.OK) as dlg:
            dlg.ShowModal()

    def on_text_changed_cbo_screenshot(self, event):
        self.update_monitor_index()
        self.update_window_names()

    def update_monitor_index(self, *args, **kwargs):
        if self.cbo_screenshot.GetValue() == "wc_mp":
            self.lbl_monitor_index.Show()
            self.cbo_monitor_index.Show()
        else:
            self.lbl_monitor_index.Hide()
            self.cbo_monitor_index.Hide()

        self.GetSizer().Layout()

    def update_window_names(self, *args, **kwargs):
        if self.cbo_screenshot.GetValue() == "wc_mp":
            self.lbl_window_name.Show()
            self.cbo_window_name.Show()
            self.btn_reload_window_name.Show()
            self.lbl_crop_region.Show()
            self.txt_crop_top.Show()
            self.lbl_crop_top.Show()
            self.txt_crop_left.Show()
            self.lbl_crop_left.Show()
            self.txt_crop_right.Show()
            self.lbl_crop_right.Show()
            self.txt_crop_bottom.Show()
            self.lbl_crop_bottom.Show()
            self.cbo_window_name.SetItems([""] + enum_window_names())
        else:
            self.lbl_window_name.Hide()
            self.cbo_window_name.Hide()
            self.btn_reload_window_name.Hide()
            self.lbl_crop_region.Hide()
            self.txt_crop_top.Hide()
            self.lbl_crop_top.Hide()
            self.txt_crop_left.Hide()
            self.lbl_crop_left.Hide()
            self.txt_crop_right.Hide()
            self.lbl_crop_right.Hide()
            self.txt_crop_bottom.Hide()
            self.lbl_crop_bottom.Hide()

        self.GetSizer().Layout()

    def on_text_changed_cbo_view_mode(self, event):
        self.update_view_mode()

    def update_view_mode(self, *args, **kwargs):
        local_viewer = self.cbo_view_mode.GetValue() == "Local Viewer"
        streaming_options = [
            self.chk_bind_addr,
            self.txt_bind_addr,
            self.btn_detect_ip,
            self.lbl_port,
            self.txt_port,
            self.lbl_stream_quality,
            self.cbo_stream_quality,
            self.chk_gpu_jpeg,
            self.sep_network1,
            self.chk_auth,
            self.lbl_auth_username,
            self.txt_auth_username,
            self.lbl_auth_password,
            self.txt_auth_password,
        ]
        if local_viewer:
            for control in streaming_options:
                control.Hide()
        else:
            for control in streaming_options:
                control.Show()

        self.GetSizer().Layout()
        self.Fit()


LOCAL_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(locale.getdefaultlocale()[0], {})


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
