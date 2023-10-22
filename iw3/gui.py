import nunif.pythonw_fix  # noqa
import locale
import sys
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
from wx.lib.buttons import GenBitmapButton
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_text, is_video, is_output_dir, make_output_filename,
    has_rembg_model)
from . import zoedepth_model as ZU
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS
from nunif.utils.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    resolve_default_dir, extension_list_to_wildcard, validate_number,
    set_icon_ex, start_file, load_icon)
from .locales import LOCALES
from . import models # noqa
import torch


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
CONFIG_PATH = path.join(path.dirname(__file__), "..", "tmp", "iw3-gui.cfg")
os.makedirs(path.dirname(CONFIG_PATH), exist_ok=True)


LAYOUT_DEBUG = False


class IW3App(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker(main_frame.GetTitle())
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(None,
                                  message=(T("Another instance is running") + "\n"
                                           + T("Are you sure you want to do this?")),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
        self.SetTopWindow(main_frame)
        return True


class MainFrame(wx.Frame):
    def __init__(self):
        super(MainFrame, self).__init__(
            None,
            name="iw3-gui",
            title=T("iw3-gui"),
            size=(1000, 720),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.initialize_component()

    def initialize_component(self):
        self.SetFont(wx.Font(10, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.CreateStatusBar()

        # input output panel

        self.pnl_file = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_file.SetBackgroundColour("#ccf")

        self.lbl_input = wx.StaticText(self.pnl_file, label=T("Input"))
        self.txt_input = wx.TextCtrl(self.pnl_file, name="txt_input")
        self.btn_input_file = GenBitmapButton(self.pnl_file, bitmap=load_icon("image-open.png"))
        self.btn_input_file.SetToolTip(T("Choose a file"))
        self.btn_input_dir = GenBitmapButton(self.pnl_file, bitmap=load_icon("folder-open.png"))
        self.btn_input_dir.SetToolTip(T("Choose a directory"))
        self.btn_input_play = GenBitmapButton(self.pnl_file, bitmap=load_icon("media-playback-start.png"))
        self.btn_input_play.SetToolTip(T("Play"))

        self.lbl_output = wx.StaticText(self.pnl_file, label=T("Output"))
        self.txt_output = wx.TextCtrl(self.pnl_file, name="txt_output")
        self.btn_same_output_dir = GenBitmapButton(self.pnl_file, bitmap=load_icon("emblem-symbolic-link.png"))
        self.btn_same_output_dir.SetToolTip(T("Set the same directory"))
        self.btn_output_dir = GenBitmapButton(self.pnl_file, bitmap=load_icon("folder-open.png"))
        self.btn_output_dir.SetToolTip(T("Choose a directory"))
        self.btn_output_play = GenBitmapButton(self.pnl_file, bitmap=load_icon("media-playback-start.png"))
        self.btn_output_play.SetToolTip(T("Play"))

        self.chk_resume = wx.CheckBox(self.pnl_file, label=T("Resume"), name="chk_resume")
        self.chk_resume.SetToolTip(T("Skip processing when the output file already exists"))
        self.chk_resume.SetValue(True)

        self.chk_recursive = wx.CheckBox(self.pnl_file, label=T("Process all subfolders"),
                                         name="chk_recursive")
        self.chk_recursive.SetValue(False)

        sublayout = wx.BoxSizer(wx.HORIZONTAL)
        sublayout.Add(self.chk_resume, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.chk_recursive, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_input, (0, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_input, (0, 1), flag=wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        layout.Add(self.btn_input_file, (0, 2), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_input_dir, (0, 3), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_input_play, (0, 4), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)

        layout.Add(self.lbl_output, (1, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_output, (1, 1), flag=wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        layout.Add(self.btn_same_output_dir, (1, 2), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_output_dir, (1, 3), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_output_play, (1, 4), flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(sublayout, (2, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

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
        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(1)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence = wx.ComboBox(self.grp_stereo, choices=["0.0", "0.5", "1.0"],
                                           style=wx.CB_DROPDOWN, name="cbo_convergence")
        self.cbo_convergence.SetSelection(1)
        self.cbo_convergence.SetToolTip("Convergence")

        self.lbl_ipd_offset = wx.StaticText(self.grp_stereo, label=T("Your Own Size"))
        # SpinCtrlDouble is better, but cannot save with PersistenceManager
        self.sld_ipd_offset = wx.SpinCtrl(self.grp_stereo, value="0", min=-10, max=20, name="sld_ipd_offset")
        self.sld_ipd_offset.SetToolTip("IPD Offset")

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo, choices=["row_flow", "grid_sample"],
                                      style=wx.CB_READONLY, name="cbo_method")
        self.cbo_method.SetSelection(0)

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo, choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
                                           style=wx.CB_READONLY, name="cbo_depth_model")
        self.cbo_depth_model.SetSelection(0)

        self.lbl_mapper = wx.StaticText(self.grp_stereo, label=T("Depth Mapping"))
        self.cbo_mapper = wx.ComboBox(self.grp_stereo,
                                      choices=["pow2", "softplus", "softplus2", "none"],
                                      style=wx.CB_READONLY, name="cbo_mapper")
        self.cbo_mapper.SetSelection(0)

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(self.grp_stereo, choices=["Full SBS", "Half SBS", "VR90"],
                                             style=wx.CB_READONLY, name="cbo_stereo_format")
        self.cbo_stereo_format.SetSelection(0)

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
                                             label=T("Flicker Reduction") + " " + T("(experimental)"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetToolTip(T("Video Only"))

        layout = wx.FlexGridSizer(rows=8, cols=2, vgap=4, hgap=4)
        layout.Add(self.lbl_divergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, 1, wx.EXPAND)
        layout.Add(self.lbl_convergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence, 1, wx.EXPAND)
        layout.Add(self.lbl_ipd_offset, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_ipd_offset, 1, wx.EXPAND)
        layout.Add(self.lbl_method, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, 1, wx.EXPAND)
        layout.Add(self.lbl_depth_model, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_model, 1, wx.EXPAND)
        layout.Add(self.lbl_mapper, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_mapper, 1, wx.EXPAND)
        layout.Add(self.lbl_stereo_format, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, 1, wx.EXPAND)
        layout.Add(self.chk_ema_normalize, 1, wx.EXPAND)

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
        self.chk_start_time = wx.CheckBox(self.grp_video_filter, label=T("Start Time"),
                                          name="chk_start_time")
        self.txt_start_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                       name="txt_start_time")
        self.chk_end_time = wx.CheckBox(self.grp_video_filter, label=T("End Time"), name="chk_end_time")
        self.txt_end_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                     name="txt_end_time")

        self.lbl_deinterlace = wx.StaticText(self.grp_video_filter, label=T("Deinterlace"))
        self.cbo_deinterlace = wx.ComboBox(self.grp_video_filter, choices=["", "yadif"],
                                           style=wx.CB_READONLY, name="cbo_deinterlace")
        self.cbo_deinterlace.SetSelection(0)

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")

        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=(200, -1),
                                      style=wx.CB_READONLY, name="cbo_rotate")
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)

        self.lbl_pad = wx.StaticText(self.grp_video_filter, label=T("Padding"))
        self.cbo_pad = wx.ComboBox(self.grp_video_filter, choices=["", "1", "2"],
                                   style=wx.CB_DROPDOWN, name="cbo_pad")
        self.cbo_pad.SetSelection(0)

        self.lbl_max_output_size = wx.StaticText(self.grp_video_filter, label=T("Output Size Limit"))
        self.cbo_max_output_size = wx.ComboBox(self.grp_video_filter,
                                               choices=["",
                                                        "1920x1080", "1280x720", "640x360",
                                                        "1080x1920", "720x1280", "360x640"],
                                               style=wx.CB_READONLY, name="cbo_max_output_size")
        self.cbo_max_output_size.SetSelection(0)

        self.chk_keep_aspect_ratio = wx.CheckBox(self.grp_video_filter, label=T("Keep Aspect Ratio"),
                                                 name="chk_keep_aspect_ratio")
        self.chk_keep_aspect_ratio.SetValue(False)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_start_time, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (0, 1), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (1, 1), flag=wx.EXPAND)

        layout.Add(self.lbl_deinterlace, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (4, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pad, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pad, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_max_output_size, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_output_size, (6, 1), flag=wx.EXPAND)
        layout.Add(self.chk_keep_aspect_ratio, (7, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA, Low VRAM, fp16
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=(200, -1), style=wx.CB_READONLY,
                                      name="cbo_device")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_device.Append(device_name, i)
            if torch.cuda.device_count() > 0:
                 self.cbo_device.Append(T("All CUDA Device"), -2)
        elif torch.backends.mps.is_available():
            self.cbo_device.Append("MPS", 0)
        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_zoed_resolution = wx.StaticText(self.grp_processor, label=T("Depth") + " " + T("Resolution"))
        self.cbo_zoed_resolution = wx.ComboBox(self.grp_processor,
                                               choices=["Default", "512"],
                                               style=wx.CB_READONLY, name="cbo_zoed_resolution")
        self.cbo_zoed_resolution.SetSelection(0)
        self.zoed_resolution = [None, 512]
        self.lbl_zoed_batch_size = wx.StaticText(self.grp_processor, label=T("Depth") + " " + T("Batch Size"))
        self.cbo_zoed_batch_size = wx.ComboBox(self.grp_processor,
                                               choices=[str(n) for n in (64, 32, 16, 8, 4, 2, 1)],
                                               style=wx.CB_READONLY, name="cbo_zoed_batch_size")
        self.cbo_zoed_batch_size.SetToolTip(T("Video Only"))
        self.cbo_zoed_batch_size.SetSelection(5)
        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Stereo") + " " + T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor,
                                          choices=[str(n) for n in (128, 64, 32, 16, 8, 4)],
                                          style=wx.CB_READONLY, name="cbo_batch_size")
        self.cbo_batch_size.SetSelection(3)

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(T("Use flip augmentation to improve depth quality (slow)"))
        self.chk_fp16 = wx.CheckBox(self.grp_processor, label=T("FP16"), name="chk_fp16")
        self.chk_fp16.SetToolTip(T("Use FP16 (fast)"))
        self.chk_fp16.SetValue(True)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_zoed_resolution, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_zoed_resolution, (1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_zoed_batch_size, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_zoed_batch_size, (2, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (3, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (4, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (4, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (4, 2), flag=wx.EXPAND)

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
        self.btn_input_play.Bind(wx.EVT_BUTTON, self.on_click_btn_input_play)
        self.btn_output_dir.Bind(wx.EVT_BUTTON, self.on_click_btn_output_dir)
        self.btn_same_output_dir.Bind(wx.EVT_BUTTON, self.on_click_btn_same_output_dir)
        self.btn_output_play.Bind(wx.EVT_BUTTON, self.on_click_btn_output_play)

        self.txt_input.Bind(wx.EVT_TEXT, self.on_text_changed_txt_input)
        self.txt_output.Bind(wx.EVT_TEXT, self.on_text_changed_txt_output)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.SetDropTarget(FileDropCallback(self.on_drop_files))
        # Disable default drop target
        for control in (self.txt_input, self.txt_output, self.txt_vf,
                        self.cbo_divergence, self.cbo_convergence, self.cbo_pad):
            control.SetDropTarget(FileDropCallback(self.on_drop_files))

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
        self.update_input_option_state()

    def on_close(self, event):
        self.persistence_manager.SaveAndUnregister()
        event.Skip()

    def on_drop_files(self, x, y, filenames):
        if filenames:
            self.txt_input.SetValue(filenames[0])
            if not self.txt_output.GetValue():
                self.set_same_output_dir()
        return True

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

    def update_input_option_state(self):
        input_path = self.txt_input.GetValue()
        if path.isdir(input_path) or is_text(input_path):
            self.chk_resume.Enable()
            self.chk_recursive.Enable()
        else:
            self.chk_resume.Disable()
            self.chk_recursive.Disable()
        self.chk_recursive.SetValue(False)

    def reset_time_range(self):
        self.chk_start_time.SetValue(False)
        self.chk_end_time.SetValue(False)
        self.txt_start_time.SetValue("00:00:00")
        self.txt_end_time.SetValue("00:00:00")

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

    def on_click_btn_input_play(self, event):
        start_file(self.txt_input.GetValue())

    def on_click_btn_output_play(self, event):
        input_path = self.txt_input.GetValue()
        output_path = self.txt_output.GetValue()
        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        video = is_video(input_path)

        if is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, video=video, vr180=vr180, half_sbs=half_sbs))

        if path.exists(output_path):
            start_file(output_path)
        elif path.exists(path.dirname(output_path)):
            start_file(path.dirname(output_path))

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
        self.update_input_option_state()
        self.reset_time_range()

    def on_text_changed_txt_output(self, event):
        self.update_start_button_state()

    def confirm_overwrite(self):
        input_path = self.txt_input.GetValue()
        output_path = self.txt_output.GetValue()
        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        video = is_video(input_path)

        if is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, video=video, vr180=vr180, half_sbs=half_sbs))
        else:
            output_path = output_path

        if path.exists(output_path):
            with wx.MessageDialog(None,
                                  message=output_path + "\n" + T("already exists. Overwrite?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                return dlg.ShowModal() == wx.ID_YES
        else:
            return True

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
                None,
                message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
                caption=T("Error"),
                style=wx.OK) as dlg:
            dlg.ShowModal()

    def on_click_btn_start(self, event):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 6.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 6.0)
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
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
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
        if device_id == -2:
            # All CUDA
            device_id = list(range(torch.cuda.device_count()))
        else:
            device_id = [device_id]

        depth_model_type = self.cbo_depth_model.GetValue()
        zoed_height = self.zoed_resolution[self.cbo_zoed_resolution.GetSelection()]
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id or
                                         self.depth_model_height != zoed_height)):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ZU.has_model(depth_model_type):
                # Realod depth model
                self.SetStatusText(f"Loading {depth_model_type}...")
            else:
                # Need to download the model
                self.SetStatusText(f"Downloading {depth_model_type}...")

        remove_bg = self.chk_rembg.GetValue()
        bg_model_type = self.cbo_bg_model.GetValue()
        if remove_bg and not has_rembg_model(bg_model_type):
            self.SetStatusText(f"Downloading {bg_model_type}...")

        max_output_width = max_output_height = None
        max_output_size = self.cbo_max_output_size.GetValue()
        if max_output_size:
            max_output_width, max_output_height = [int(s) for s in max_output_size.split("x")]

        input_path = self.txt_input.GetValue()
        resume = (path.isdir(input_path) or is_text(input_path)) and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None

        parser.set_defaults(
            input=input_path,
            output=self.txt_output.GetValue(),
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            ipd_offset=float(self.sld_ipd_offset.GetValue()),
            method=self.cbo_method.GetValue(),
            depth_model=depth_model_type,
            mapper=self.cbo_mapper.GetValue(),
            vr180=vr180,
            half_sbs=half_sbs,
            ema_normalize=self.chk_ema_normalize.GetValue(),

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
            max_output_width=max_output_width,
            max_output_height=max_output_height,
            keep_aspect_ratio=self.chk_keep_aspect_ratio.GetValue(),

            gpu=device_id,
            batch_size=int(self.cbo_batch_size.GetValue()),
            zoed_batch_size=int(self.cbo_zoed_batch_size.GetValue()),
            zoed_height=zoed_height,
            tta=self.chk_tta.GetValue(),
            disable_amp=not self.chk_fp16.GetValue(),
            low_vram=self.chk_low_vram.GetValue(),

            resume=resume,
            recursive=recursive,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            tqdm_fn=functools.partial(TQDMGUI, self),
            depth_model=self.depth_model)
        startWorker(self.on_exit_worker, iw3_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            args = result.get()
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.zoed_height

            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
            else:
                self.SetStatusText(T("Cancelled"))
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.update_start_button_state()

        # free vram
        gc.collect()
        if torch.cuda.is_available:
            torch.cuda.empty_cache()

    def on_click_btn_cancel(self, event):
        self.stop_event.set()

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            self.prg_tqdm.SetRange(value)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.SetStatusText(f"{0}/{value} {desc}")
        elif type == 1:
            # update
            if self.prg_tqdm.GetValue() + value <= self.prg_tqdm.GetRange():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            else:
                self.prg_tqdm.SetRange(self.prg_tqdm.GetValue() + value)
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
            self.SetStatusText(f"{pos}/{end_pos} [ {t}, {fps:.2f}FPS ] {desc}")
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

    app = IW3App()
    app.MainLoop()


if __name__ == "__main__":
    main()
