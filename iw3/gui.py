import nunif.pythonw_fix  # noqa
import locale
import os
from os import path
import gc
import traceback
import functools
from time import time
import threading
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as wxpm
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_video, is_output_dir, make_output_filename,
    load_depth_model, has_depth_model, has_rembg_model,
)
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS
from .locales import LOCALES
from . import models # noqa
import torch


IMAGE_EXTENSIONS = ";".join(["*" + ext for ext in LOADER_SUPPORTED_EXTENSIONS])
VIDEO_EXTENSIONS = ";".join(["*" + ext for ext in KNOWN_VIDEO_EXTENSIONS])
VIDEO_EXTENSIONS = "*.mp4;*.mkv;*.mpeg;*.mpg;*.avi;*.wmv;*.ogg;*.ts;*.mov;*.flv;*.webm"
CONFIG_PATH = path.join(path.dirname(__file__), "..", "tmp", "iw3-gui.cfg")
os.makedirs(path.dirname(CONFIG_PATH), exist_ok=True)


def resolve_default_dir(src):
    if src:
        if "." in path.basename(src):
            default_dir = path.dirname(src)
        else:
            default_dir = src
    else:
        default_dir = ""
    return default_dir


def validate_number(s, min_value, max_value, is_int=False, allow_empty=False):
    if allow_empty and (s is None or s == ""):
        return True
    try:
        if is_int:
            v = int(s)
        else:
            v = float(s)
        return min_value <= v and v <= max_value
    except ValueError:
        return False


myEVT_TQDM = wx.NewEventType()
EVT_TQDM = wx.PyEventBinder(myEVT_TQDM, 1)


class TQDMEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, type=None, value=None):
        super(TQDMEvent, self).__init__(etype, eid)
        self.type = type
        self.value = value

    def GetValue(self):
        return (self.type, self.value)


class TQDMGUI():
    def __init__(self, parent, **kwargs):
        self.parent = parent
        total = kwargs["total"]
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 0, total))

    def update(self, n=1):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 1, n))

    def close(self):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 2, 0))


LAYOUT_DEBUG = False


class MainFrame(wx.Frame):
    def __init__(self):
        super(MainFrame, self).__init__(
            None,
            name="iw3-gui",
            title=T("iw3-gui"),
            size=(940, 580),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.initialize_component()

    def initialize_component(self):
        ICON_BUTTON_SIZE = (-1, -1)
        self.SetFont(wx.Font(10, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.CreateStatusBar()

        # input output panel

        self.pnl_file = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_file.SetBackgroundColour("#ccf")

        self.lbl_input = wx.StaticText(self.pnl_file, label=T("Input"))
        self.lbl_output = wx.StaticText(self.pnl_file, label=T("Output"))
        self.txt_input = wx.TextCtrl(self.pnl_file, name="txt_input")
        self.txt_output = wx.TextCtrl(self.pnl_file, name="txt_output")
        self.btn_input_file = wx.Button(self.pnl_file, label=T("..."),
                                        size=ICON_BUTTON_SIZE, style=wx.BU_EXACTFIT)
        self.btn_input_file.SetToolTip(T("Choose a file"))
        self.btn_input_dir = wx.Button(self.pnl_file, label=T("..."),
                                       size=ICON_BUTTON_SIZE, style=wx.BU_EXACTFIT)
        self.btn_input_dir.SetToolTip(T("Choose a directory"))
        self.btn_same_output_dir = wx.Button(self.pnl_file, label=T("<<<"),
                                             size=ICON_BUTTON_SIZE, style=wx.BU_EXACTFIT)
        self.btn_same_output_dir.SetToolTip(T("Set the same directory"))
        self.btn_output_dir = wx.Button(self.pnl_file, label=T("..."),
                                        size=ICON_BUTTON_SIZE, style=wx.BU_EXACTFIT)
        self.btn_output_dir.SetToolTip(T("Choose a directory"))

        layout = wx.FlexGridSizer(rows=2, cols=4, vgap=4, hgap=4)
        layout.Add(self.lbl_input, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_input, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        layout.Add(self.btn_input_file, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_input_dir, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)

        layout.Add(self.lbl_output, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_output, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        layout.Add(self.btn_same_output_dir, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_output_dir, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.AddGrowableCol(1)
        self.pnl_file.SetSizer(layout)

        # options panel

        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # stereo generation settings
        # divergence, convergence, method, depth_model, mapper

        self.grp_stereo = wx.StaticBox(self.pnl_options, label=T("Stereo Generation"))

        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = wx.ComboBox(self.grp_stereo, choices=["2.5", "2.0", "1.0"],
                                          style=wx.CB_DROPDOWN, name="cbo_divergence")
        self.cbo_divergence.SetSelection(1)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence = wx.ComboBox(self.grp_stereo, choices=["0.0", "0.5", "1.0"],
                                           style=wx.CB_DROPDOWN, name="cbo_convergence")
        self.cbo_convergence.SetSelection(1)

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo, choices=["row_flow", "grid_sample"],
                                      style=wx.CB_READONLY, name="cbo_method")
        self.cbo_method.SetSelection(0)

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo, choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
                                           style=wx.CB_READONLY, name="cbo_depth_model")
        self.cbo_depth_model.SetSelection(0)

        self.lbl_mapper = wx.StaticText(self.grp_stereo, label=T("Depth Mapping"))
        self.cbo_mapper = wx.ComboBox(self.grp_stereo, choices=["pow2", "softplus", "softplus2", "none"],
                                      style=wx.CB_READONLY, name="cbo_mapper")
        self.cbo_mapper.SetSelection(0)

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(self.grp_stereo, choices=["3D SBS", "VR90"],
                                             style=wx.CB_READONLY, name="cbo_stereo_format")
        self.cbo_stereo_format.SetSelection(0)

        layout = wx.FlexGridSizer(rows=6, cols=2, vgap=4, hgap=4)
        layout.Add(self.lbl_divergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, 1, wx.EXPAND)
        layout.Add(self.lbl_convergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence, 1, wx.EXPAND)
        layout.Add(self.lbl_method, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, 1, wx.EXPAND)
        layout.Add(self.lbl_depth_model, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_model, 1, wx.EXPAND)
        layout.Add(self.lbl_mapper, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_mapper, 1, wx.EXPAND)
        layout.Add(self.lbl_stereo_format, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, 1, wx.EXPAND)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # video encoding
        # sbs/vr180, padding
        # max-fps, crf, preset, tune
        self.grp_video = wx.StaticBox(self.pnl_options, label=T("Video Encoding"))

        self.lbl_fps = wx.StaticText(self.grp_video, label=T("Max FPS"))
        self.cbo_fps = wx.ComboBox(self.grp_video, choices=["0.25", "1", "15", "30", "60", "1000"],
                                   style=wx.CB_READONLY, name="cbo_fps")
        self.cbo_fps.SetSelection(3)

        self.lbl_crf = wx.StaticText(self.grp_video, label=T("CRF"))
        self.cbo_crf = wx.ComboBox(self.grp_video, choices=[str(n) for n in range(16, 28)],
                                   style=wx.CB_READONLY, name="cbo_crf")
        self.cbo_crf.SetSelection(4)

        self.lbl_preset = wx.StaticText(self.grp_video, label=T("Preset"))
        self.cbo_preset = wx.ComboBox(
            self.grp_video, choices=[
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow", "placebo"],
            style=wx.CB_READONLY, name="cbo_preset")
        self.cbo_preset.SetSelection(0)

        self.lbl_tune = wx.StaticText(self.grp_video, label=T("Tune"))
        self.cbo_tune = wx.ComboBox(
            self.grp_video, choices=["", "film", "animation", "grain", "stillimage", "psnr"],
            style=wx.CB_READONLY, name="cbo_tune")
        self.cbo_tune.SetSelection(0)
        self.chk_tune_fastdecode = wx.CheckBox(self.grp_video, label=T("fastdecode"),
                                               name="chk_tune_fastdecode")
        self.chk_tune_zerolatency = wx.CheckBox(self.grp_video, label=T("zerolatency"),
                                                name="chk_tune_zerolatency")
        self.chk_tune_zerolatency.SetValue(True)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_fps, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_fps, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_crf, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_crf, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_preset, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_preset, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tune, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tune, (3, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_fastdecode, (4, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_zerolatency, (5, 1), flag=wx.EXPAND)

        sizer_video = wx.StaticBoxSizer(self.grp_video, wx.VERTICAL)
        sizer_video.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # background removal
        self.grp_rembg = wx.StaticBox(self.pnl_options, label=T("Background Removal"))
        self.chk_rembg = wx.CheckBox(self.grp_rembg, label=T("Enable"), name="chk_rembg")
        self.lbl_bg_model = wx.StaticText(self.grp_rembg, label=T("Seg Model"))
        self.cbo_bg_model = wx.ComboBox(self.grp_rembg,
                                        choices=["u2net", "u2net_human_seg",
                                                 "isnet-general-use", "isnet-anime"],
                                        style=wx.CB_READONLY, name="cbo_bg_model")
        self.cbo_bg_model.SetSelection(1)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_rembg, (0, 0), (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_bg_model, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_bg_model, (1, 1), flag=wx.EXPAND)
        sizer_rembg = wx.StaticBoxSizer(self.grp_rembg, wx.VERTICAL)
        sizer_rembg.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # input video filter
        # deinterlace, rotate, vf
        self.grp_video_filter = wx.StaticBox(self.pnl_options, label=T("Video Filter"))
        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=(200, -1),
                                      style=wx.CB_READONLY, name="cbo_rotate")
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)
        self.lbl_deinterlace = wx.StaticText(self.grp_video_filter, label=T("Deinterlace"))
        self.cbo_deinterlace = wx.ComboBox(self.grp_video_filter, choices=["", "yadif"],
                                           style=wx.CB_READONLY, name="cbo_deinterlace")
        self.cbo_deinterlace.SetSelection(0)

        self.lbl_pad = wx.StaticText(self.grp_video_filter, label=T("Padding"))
        self.cbo_pad = wx.ComboBox(self.grp_video_filter, choices=["", "1", "2"],
                                   style=wx.CB_DROPDOWN, name="cbo_pad")
        self.cbo_pad.SetSelection(0)

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_deinterlace, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pad, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pad, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (3, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA, Low VRAM
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=(240, -1), style=wx.CB_READONLY,
                                      name="cbo_device")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_device.Append(device_name, i)
        elif torch.backends.mps.is_available():
            self.cbo_device.Append("MPS", 0)
        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor, style=wx.CB_READONLY,
                                          name="cbo_batch_size")
        for n in (64, 32, 16, 8, 4):
            self.cbo_batch_size.Append(str(n), n)
        self.cbo_batch_size.SetSelection(1)

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (1, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (2, 0), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (2, 1), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        layout = wx.GridBagSizer(wx.HORIZONTAL)
        layout.Add(sizer_stereo, (0, 0), (2, 0), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video, (0, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_rembg, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video_filter, (0, 2), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, (1, 2), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(layout)

        # processing panel
        self.pnl_process = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_process.SetBackgroundColour("#fcc")
        self.prg_tqdm = wx.Gauge(self.pnl_process, style=wx.GA_HORIZONTAL)
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Cancel"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.prg_tqdm, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        layout.Add(self.btn_start, 0, wx.ALL, 4)
        layout.Add(self.btn_cancel, 0, wx.ALL, 4)
        self.pnl_process.SetSizer(layout)

        # main layout

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.pnl_file, 0, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_options, 1, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_process, 0, wx.ALL | wx.EXPAND, 8)
        self.SetSizer(layout)

        # bind
        self.btn_input_file.Bind(wx.EVT_BUTTON, self.on_click_btn_input_file)
        self.btn_input_dir.Bind(wx.EVT_BUTTON, self.on_click_btn_input_dir)
        self.btn_output_dir.Bind(wx.EVT_BUTTON, self.on_click_btn_output_dir)
        self.btn_same_output_dir.Bind(wx.EVT_BUTTON, self.on_click_btn_same_output_dir)

        self.txt_input.Bind(wx.EVT_TEXT, self.on_text_changed_txt_input)
        self.txt_output.Bind(wx.EVT_TEXT, self.on_text_changed_txt_output)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_file.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()

        self.persistence_manager = wxpm.PersistenceManager.Get()
        self.persistence_manager.SetManagerStyle(
            wxpm.PM_DEFAULT_STYLE | wxpm.PM_PERSIST_CONTROL_VALUE | wxpm.PM_SAVE_RESTORE_TREE_LIST_SELECTIONS)
        self.persistence_manager.SetPersistenceFile(CONFIG_PATH)
        self.persistence_manager.RegisterAndRestoreAll(self)
        self.persistence_manager.Save(self)

        self.update_start_button_state()
        self.update_rembg_state()

    def on_close(self, event):
        self.persistence_manager.SaveAndUnregister()
        event.Skip()

    def update_start_button_state(self):
        if not self.processing:
            if self.txt_input.GetValue() and self.txt_output.GetValue():
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()

    def update_rembg_state(self):
        if is_video(self.txt_input.GetValue()):
            self.chk_rembg.SetValue(False)
            self.chk_rembg.Disable()
            self.cbo_bg_model.Disable()
        else:
            self.chk_rembg.Enable()
            self.cbo_bg_model.Enable()

    def set_same_output_dir(self):
        selected_path = self.txt_input.GetValue()
        if path.isdir(selected_path):
            self.txt_output.SetValue(path.join(selected_path, "iw3"))
        else:
            self.txt_output.SetValue(path.join(path.dirname(selected_path), "iw3"))

    def on_click_btn_input_file(self, event):
        wildcard = (f"Image and Video files|{IMAGE_EXTENSIONS};{VIDEO_EXTENSIONS}"
                    f"|Video files|{VIDEO_EXTENSIONS}"
                    f"|Image files|{IMAGE_EXTENSIONS}"
                    "|All Files|*.*")
        default_dir = resolve_default_dir(self.txt_input.GetValue())
        with wx.FileDialog(self.pnl_file, T("Choose a file"),
                           wildcard=wildcard, defaultDir=default_dir,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg_file:
            if dlg_file.ShowModal() == wx.ID_CANCEL:
                return
            selected_path = dlg_file.GetPath()
            self.txt_input.SetValue(selected_path)
            if not self.txt_output.GetValue():
                self.set_same_output_dir()

    def on_click_btn_input_dir(self, event):
        default_dir = resolve_default_dir(self.txt_input.GetValue())
        with wx.DirDialog(self.pnl_file, T("Choose a directory"),
                          defaultPath=default_dir,
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg_dir:
            if dlg_dir.ShowModal() == wx.ID_CANCEL:
                return
            selected_path = dlg_dir.GetPath()
            self.txt_input.SetValue(selected_path)
            if not self.txt_output.GetValue():
                self.set_same_output_dir()

    def on_click_btn_same_output_dir(self, event):
        self.set_same_output_dir()

    def on_click_btn_output_dir(self, event):
        default_dir = resolve_default_dir(self.txt_output.GetValue())
        if not path.exists(default_dir):
            default_dir = path.dirname(default_dir)
        with wx.DirDialog(self.pnl_file, T("Choose a directory"),
                          defaultPath=default_dir,
                          style=wx.DD_DEFAULT_STYLE) as dlg_dir:
            if dlg_dir.ShowModal() == wx.ID_CANCEL:
                return
            self.txt_output.SetValue(dlg_dir.GetPath())

    def on_text_changed_txt_input(self, event):
        self.update_start_button_state()
        self.update_rembg_state()

    def on_text_changed_txt_output(self, event):
        self.update_start_button_state()

    def confirm_overwrite(self):
        input_path = self.txt_input.GetValue()
        output_path = self.txt_output.GetValue()
        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        video = is_video(input_path)

        if is_output_dir(output_path):
            output_path = path.join(output_path, make_output_filename(input_path, video=video, vr180=vr180))
        else:
            output_path = output_path

        if path.exists(output_path):
            ret = wx.MessageDialog(
                None,
                message=output_path + "\n" + T("already exists. Overwrite?"),
                caption=T("Confirm"),
                style=wx.YES_NO).ShowModal()
            return ret == wx.ID_YES
        else:
            return True

    def show_validation_error_message(self, name, min_value, max_value):
        wx.MessageDialog(
            None,
            message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
            caption=T("Error"),
            style=wx.OK).ShowModal()

    def on_click_btn_start(self, event):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 2.5):
            self.show_validation_error_message(T("3D Strength"), 0.0, 2.5)
            return
        if not validate_number(self.cbo_convergence.GetValue(), 0.0, 1.0):
            self.show_validation_error_message(T("Convergence Plane"), 0.0, 1.0)
            return
        if not validate_number(self.cbo_pad.GetValue(), 0.0, 10.0, allow_empty=True):
            self.show_validation_error_message(T("Padding"), 0.0, 10.0)
            return
        if not self.confirm_overwrite():
            return

        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.stop_event.clear()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        parser = create_parser(required_true=False)
        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        tune = set()
        if self.chk_tune_zerolatency.GetValue():
            tune.add("zerolatency")
        if self.chk_tune_fastdecode.GetValue():
            tune.add("fastdecode")
        if self.cbo_tune.GetValue():
            tune.add(self.cbo_tune.GetValue())
        if self.cbo_pad.GetValue():
            pad = float(self.cbo_pad.GetValue())
        else:
            pad = None
        rot = self.cbo_rotate.GetClientData(self.cbo_rotate.GetSelection())
        rotate_left = rotate_right = None
        if rot == "left":
            rotate_left = True
        elif rot == "right":
            rotate_right = True

        vf = []
        if self.cbo_deinterlace.GetValue():
            vf += [self.cbo_deinterlace.GetValue()]
        if self.txt_vf.GetValue():
            vf += [self.txt_vf.GetValue()]
        vf = ",".join(vf)

        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        depth_model_type = self.cbo_depth_model.GetValue()
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id)):
            if has_depth_model(depth_model_type):
                # Realod depth model
                self.SetStatusText(f"Loading {depth_model_type}...")
                self.depth_model = None
                gc.collect()
                torch.cuda.empty_cache()
                self.depth_model = load_depth_model(model_type=depth_model_type, gpu=device_id)
                self.depth_model_type = depth_model_type
                self.depth_model_device_id = device_id
            else:
                # Need to download the model, so download the model in the worker thread
                self.SetStatusText(f"Downloading {depth_model_type}...")
                self.depth_model = None
                self.depth_model_type = None
                self.depth_model_device_id = None

        remove_bg = self.chk_rembg.GetValue()
        bg_model_type = self.cbo_bg_model.GetValue()
        if remove_bg and not has_rembg_model(bg_model_type):
            self.SetStatusText(f"Downloading {bg_model_type}...")

        parser.set_defaults(
            input=self.txt_input.GetValue(),
            output=self.txt_output.GetValue(),
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            method=self.cbo_method.GetValue(),
            depth_model=depth_model_type,
            mapper=self.cbo_mapper.GetValue(),
            vr180=vr180,

            max_fps=float(self.cbo_fps.GetValue()),
            crf=int(self.cbo_crf.GetValue()),
            preset=self.cbo_preset.GetValue(),
            tune=list(tune),

            remove_bg=remove_bg,
            bg_model=bg_model_type,

            pad=pad,
            rotate_right=rotate_right,
            rotate_left=rotate_left,
            vf=vf,

            gpu=device_id,
            batch_size=int(self.cbo_batch_size.GetValue()),
            tta=self.chk_tta.GetValue(),
            disable_zoedepth_batch=self.chk_low_vram.GetValue(),
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            tqdm_fn=functools.partial(TQDMGUI, self),
            depth_model=self.depth_model,
        )
        startWorker(self.on_exit_worker, iw3_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            ret = result.get()  # noqa
            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
            else:
                self.SetStatusText(T("Cancelled"))
        except:  # noqa
            self.SetStatusText(T("Error"))
            message = traceback.format_exc()
            if len(message) > 1024:
                message = "..." + message[-1024:]
            wx.MessageBox(message, T("Error"), wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.update_start_button_state()

        # free vram
        gc.collect()
        torch.cuda.empty_cache()

    def on_click_btn_cancel(self, event):
        self.stop_event.set()

    def on_tqdm(self, event):
        type, value = event.GetValue()
        if type == 0:
            # initialize
            self.prg_tqdm.SetRange(value)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.SetStatusText(f"{0}/{value}")
        elif type == 1:
            # update
            self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            now = time()
            pos = self.prg_tqdm.GetValue()
            end_pos = self.prg_tqdm.GetRange()
            fps = pos / (now - self.start_time)
            remaining_time = int((end_pos - pos) / fps)
            h = remaining_time // 3600
            m = (remaining_time - h * 3600) // 60
            s = (remaining_time - h * 3600 - m * 60)
            t = f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"
            self.SetStatusText(f"{pos}/{end_pos} [ {t}, {fps:.2f}FPS ]")
        elif type == 2:
            # close
            pass


LOCALE_DICT = LOCALES.get(locale.getlocale()[0], {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, help="lang, ja_JP, en_US")
    args = parser.parse_args()
    if args.lang:
        global LOCALE_DICT
        LOCALE_DICT = LOCALES.get(args.lang, {})
    sys.argv = [sys.argv[0]]  # clear command arguments

    app = wx.App()
    main = MainFrame()
    main.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
