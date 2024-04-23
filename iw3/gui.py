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
import wx.lib.agw.persist as persist
from wx.lib.buttons import GenBitmapButton
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_text, is_video, is_output_dir, make_output_filename,
    has_rembg_model)
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS
from nunif.utils.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_restore_all, persistent_manager_register,
    resolve_default_dir, extension_list_to_wildcard, validate_number,
    set_icon_ex, start_file, load_icon)
from .locales import LOCALES
from . import models # noqa
from .depth_anything_model import MODEL_FILES as DEPTH_ANYTHING_MODELS
from . import export_config
import torch


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
YAML_EXTENSIONS = extension_list_to_wildcard((".yml", ".yaml"))
CONFIG_PATH = path.join(path.dirname(__file__), "..", "tmp", "iw3-gui.cfg")
os.makedirs(path.dirname(CONFIG_PATH), exist_ok=True)


LAYOUT_DEBUG = False


class IW3App(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker(main_frame.GetTitle())
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(None,
                                  message=(T("Another instance is running") + "\n" +
                                           T("Are you sure you want to do this?")),
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
            size=(1100, 720),
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
        self.cbo_divergence = EditableComboBox(self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"],
                                               name="cbo_divergence")
        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(4)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence = EditableComboBox(self.grp_stereo, choices=["0.0", "0.5", "1.0"],
                                                name="cbo_convergence")
        self.cbo_convergence.SetSelection(1)
        self.cbo_convergence.SetToolTip("Convergence")

        self.lbl_ipd_offset = wx.StaticText(self.grp_stereo, label=T("Your Own Size"))
        # SpinCtrlDouble is better, but cannot save with PersistenceManager
        self.sld_ipd_offset = wx.SpinCtrl(self.grp_stereo, value="0", min=-10, max=20, name="sld_ipd_offset")
        self.sld_ipd_offset.SetToolTip("IPD Offset")

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo, choices=["row_flow_v3_sym", "row_flow_v3_rev2", "row_flow_v3",
                                                                "row_flow_v2", "forward_fill"],
                                      style=wx.CB_READONLY, name="cbo_method")
        self.cbo_method.SetSelection(0)

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
                                           choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK",
                                                    "ZoeD_Any_N", "ZoeD_Any_K",
                                                    "Any_S", "Any_B", "Any_L"],
                                           style=wx.CB_READONLY, name="cbo_depth_model")
        self.cbo_depth_model.SetSelection(0)

        self.lbl_zoed_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_zoed_resolution = EditableComboBox(self.grp_stereo,
                                                    choices=["Default", "512"],
                                                    name="cbo_zoed_resolution")
        self.cbo_zoed_resolution.SetSelection(0)

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = wx.ComboBox(self.grp_stereo,
                                                choices=["-3", "-2", "-1", "0", "1", "2", "3"],
                                                style=wx.CB_READONLY, name="cbo_foreground_scale")
        self.cbo_foreground_scale.SetSelection(3)

        self.chk_edge_dilation = wx.CheckBox(self.grp_stereo, label=T("Edge Fix"), name="chk_edge_dilation")
        self.cbo_edge_dilation = EditableComboBox(self.grp_stereo,
                                                  choices=["0", "1", "2", "3", "4"],
                                                  name="cbo_edge_dilation")
        self.chk_edge_dilation.SetValue(False)

        self.cbo_edge_dilation.SetSelection(2)
        self.cbo_edge_dilation.SetToolTip(T("Reduce distortion of foreground and background edges"))

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(
            self.grp_stereo,
            choices=["Full SBS", "Half SBS", "VR90",
                     "Export", "Export disparity",
                     "Anaglyph dubois",
                     "Anaglyph dubois2",
                     "Anaglyph color", "Anaglyph gray",
                     "Anaglyph half-color",
                     "Anaglyph wimmer", "Anaglyph wimmer2",
                     "Debug Depth",
                     ],
            style=wx.CB_READONLY, name="cbo_stereo_format")
        self.cbo_stereo_format.SetSelection(0)

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
                                             label=T("Flicker Reduction"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))
        self.cbo_ema_decay = EditableComboBox(self.grp_stereo, choices=["0.99", "0.9", "0.75", "0.5"],
                                              name="cbo_ema_decay")
        self.cbo_ema_decay.SetSelection(2)

        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))

        layout = wx.FlexGridSizer(rows=10, cols=2, vgap=4, hgap=4)
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
        layout.Add(self.lbl_zoed_resolution, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_zoed_resolution, 1, wx.EXPAND)
        layout.Add(self.lbl_foreground_scale, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_scale, 1, wx.EXPAND)
        layout.Add(self.chk_edge_dilation, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_edge_dilation, 1, wx.EXPAND)
        layout.Add(self.chk_ema_normalize, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_ema_decay, 1, wx.EXPAND)
        layout.Add(self.lbl_stereo_format, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, 1, wx.EXPAND)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # video encoding
        # sbs/vr180, padding
        # max-fps, crf, preset, tune
        self.grp_video = wx.StaticBox(self.pnl_options, label=T("Video Encoding"))

        self.lbl_video_format = wx.StaticText(self.grp_video, label=T("Video Format"))
        self.cbo_video_format = wx.ComboBox(self.grp_video, choices=["mp4", "mkv"],
                                            style=wx.CB_READONLY, name="cbo_video_format")
        self.cbo_video_format.SetSelection(0)

        self.lbl_fps = wx.StaticText(self.grp_video, label=T("Max FPS"))
        self.cbo_fps = EditableComboBox(
            self.grp_video, choices=["1000", "60", "59.94", "30", "29.97", "24", "23.976", "15", "1", "0.25"],
            name="cbo_fps")
        self.cbo_fps.SetSelection(3)

        self.lbl_pix_fmt = wx.StaticText(self.grp_video, label=T("Pixel Format"))
        self.cbo_pix_fmt = wx.ComboBox(self.grp_video, choices=["yuv420p", "yuv444p", "rgb24"],
                                       style=wx.CB_READONLY, name="cbo_pix_fmt")
        self.cbo_pix_fmt.SetSelection(0)

        self.lbl_crf = wx.StaticText(self.grp_video, label=T("CRF"))
        self.cbo_crf = EditableComboBox(self.grp_video, choices=[str(n) for n in range(16, 28)],
                                        name="cbo_crf")
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

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_fps, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_fps, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_format, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_format, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pix_fmt, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pix_fmt, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_crf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_crf, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_preset, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_preset, (4, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tune, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tune, (5, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_fastdecode, (6, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_zerolatency, (7, 1), flag=wx.EXPAND)

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

        self.lbl_zoed_batch_size = wx.StaticText(self.grp_processor, label=T("Depth") + " " + T("Batch Size"))
        self.cbo_zoed_batch_size = wx.ComboBox(self.grp_processor,
                                               choices=[str(n) for n in (64, 32, 16, 8, 4, 2, 1)],
                                               style=wx.CB_READONLY, name="cbo_zoed_batch_size")
        self.cbo_zoed_batch_size.SetToolTip(T("Video Only"))
        self.cbo_zoed_batch_size.SetSelection(5)

        self.lbl_max_workers = wx.StaticText(self.grp_processor, label=T("Worker Threads"))
        self.cbo_max_workers = wx.ComboBox(self.grp_processor,
                                           choices=[str(n) for n in (16, 8, 4, 3, 2, 0)],
                                           style=wx.CB_READONLY, name="cbo_max_workers")
        self.cbo_max_workers.SetToolTip(T("Video Only"))
        self.cbo_max_workers.SetSelection(5)

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(T("Use flip augmentation to improve depth quality (slow)"))
        self.chk_fp16 = wx.CheckBox(self.grp_processor, label=T("FP16"), name="chk_fp16")
        self.chk_fp16.SetToolTip(T("Use FP16 (fast)"))
        self.chk_fp16.SetValue(True)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_zoed_batch_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_zoed_batch_size, (1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (2, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (3, 2), flag=wx.EXPAND)

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

        self.cbo_depth_model.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_depth_model)
        self.chk_edge_dilation.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)

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

        editable_comboxes = [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_zoed_resolution,
            self.cbo_edge_dilation,
            self.cbo_fps,
            self.cbo_crf,
        ]
        self.persistence_manager = persist.PersistenceManager.Get()
        self.persistence_manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        self.persistence_manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(self.persistence_manager, self)
        for control in editable_comboxes:
            persistent_manager_register(self.persistence_manager, control, EditableComboBoxPersistentHandler)
        persistent_manager_restore_all(self.persistence_manager)

        self.update_start_button_state()
        self.update_rembg_state()
        self.update_input_option_state()
        if not self.chk_edge_dilation.IsChecked():
            self.update_model_selection()
        self.update_edge_dilation()
        self.update_ema_normalize()

    def get_anaglyph_method(self):
        if "Anaglyph" in self.cbo_stereo_format.GetValue():
            anaglyph = self.cbo_stereo_format.GetValue().split(" ")[-1]
        else:
            anaglyph = None
        return anaglyph

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
        wildcard = (f"Image and Video and YAML files|{IMAGE_EXTENSIONS};{VIDEO_EXTENSIONS};{YAML_EXTENSIONS}"
                    f"|Video files|{VIDEO_EXTENSIONS}"
                    f"|Image files|{IMAGE_EXTENSIONS}"
                    f"|YAML files|{YAML_EXTENSIONS}"
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
        debug = self.cbo_stereo_format.GetValue() == "Debug Depth"
        anaglyph = self.get_anaglyph_method()
        video_extension = "." + self.cbo_video_format.GetValue()
        video = is_video(input_path)

        if is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, video=video,
                                     vr180=vr180, half_sbs=half_sbs, anaglyph=anaglyph,
                                     debug=debug, video_extension=video_extension))

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

    def update_model_selection(self):
        name = self.cbo_depth_model.GetValue()
        if name in DEPTH_ANYTHING_MODELS:
            self.chk_edge_dilation.SetValue(True)
            self.cbo_edge_dilation.Enable()
        else:
            self.chk_edge_dilation.SetValue(False)
            self.cbo_edge_dilation.Disable()

    def on_selected_index_changed_cbo_depth_model(self, event):
        self.update_model_selection()

    def update_edge_dilation(self):
        if self.chk_edge_dilation.IsChecked():
            self.cbo_edge_dilation.Enable()
        else:
            self.cbo_edge_dilation.Disable()

    def on_changed_chk_edge_dilation(self, event):
        self.update_edge_dilation()

    def update_ema_normalize(self):
        if self.chk_ema_normalize.IsChecked():
            self.cbo_ema_decay.Enable()
        else:
            self.cbo_ema_decay.Disable()

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()

    def confirm_overwrite(self):
        input_path = self.txt_input.GetValue()
        output_path = self.txt_output.GetValue()
        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        debug = self.cbo_stereo_format.GetValue() == "Debug Depth"
        anaglyph = self.get_anaglyph_method()
        video_extension = "." + self.cbo_video_format.GetValue()
        video = is_video(input_path)
        is_export = self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}

        if not is_export:
            if is_output_dir(output_path):
                output_path = path.join(
                    output_path,
                    make_output_filename(input_path, video=video,
                                         vr180=vr180, half_sbs=half_sbs, anaglyph=anaglyph,
                                         debug=debug, video_extension=video_extension))
            else:
                output_path = output_path
        else:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename, export_config.FILENAME)
            else:
                output_path = path.join(output_path, export_config.FILENAME)

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
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return
        if not validate_number(self.cbo_pad.GetValue(), 0.0, 10.0, allow_empty=True):
            self.show_validation_error_message(T("Padding"), 0.0, 10.0)
            return
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return
        if not validate_number(self.cbo_fps.GetValue(), 0.25, 1000.0, allow_empty=False):
            self.show_validation_error_message(T("Max FPS"), 0.25, 1000.0)
            return
        if not validate_number(self.cbo_crf.GetValue(), 0, 30, is_int=True):
            self.show_validation_error_message(T("CRF"), 0, 30)
            return
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return

        zoed_height = self.cbo_zoed_resolution.GetValue()
        if zoed_height == "Default" or zoed_height == "":
            zoed_height = None
        else:
            if not validate_number(zoed_height, 384, 2048, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 384, 2048)
                return
            zoed_height = int(zoed_height)

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
        anaglyph = self.get_anaglyph_method()
        export = self.cbo_stereo_format.GetValue() == "Export"
        export_disparity = self.cbo_stereo_format.GetValue() == "Export disparity"
        debug_depth = self.cbo_stereo_format.GetValue() == "Debug Depth"

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
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id or
                                         self.depth_model_height != zoed_height)):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        remove_bg = self.chk_rembg.GetValue()
        bg_model_type = self.cbo_bg_model.GetValue()

        max_output_width = max_output_height = None
        max_output_size = self.cbo_max_output_size.GetValue()
        if max_output_size:
            max_output_width, max_output_height = [int(s) for s in max_output_size.split("x")]

        input_path = self.txt_input.GetValue()
        resume = (path.isdir(input_path) or is_text(input_path)) and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None
        edge_dilation = int(self.cbo_edge_dilation.GetValue()) if self.chk_edge_dilation.IsChecked() else 0

        parser.set_defaults(
            input=input_path,
            output=self.txt_output.GetValue(),
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            ipd_offset=float(self.sld_ipd_offset.GetValue()),
            method=self.cbo_method.GetValue(),
            depth_model=depth_model_type,
            foreground_scale=int(self.cbo_foreground_scale.GetValue()),
            edge_dilation=edge_dilation,
            vr180=vr180,
            half_sbs=half_sbs,
            anaglyph=anaglyph,
            export=export,
            export_disparity=export_disparity,
            debug_depth=debug_depth,
            ema_normalize=self.chk_ema_normalize.GetValue(),
            ema_decay=float(self.cbo_ema_decay.GetValue()),

            max_fps=float(self.cbo_fps.GetValue()),
            pix_fmt=self.cbo_pix_fmt.GetValue(),
            video_format=self.cbo_video_format.GetValue(),
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
            zoed_batch_size=int(self.cbo_zoed_batch_size.GetValue()),
            zoed_height=zoed_height,
            max_workers=int(self.cbo_max_workers.GetValue()),
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

        if args.state["depth_utils"].has_model(depth_model_type):
            # Realod depth model
            self.SetStatusText(f"Loading {depth_model_type}...")
            if remove_bg and not has_rembg_model(bg_model_type):
                self.SetStatusText(f"Downloading {bg_model_type}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {depth_model_type}...")

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
            fps = pos / (now - self.start_time + 1e-6)
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
