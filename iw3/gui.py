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
    is_text, is_video, is_output_dir, is_yaml, make_output_filename,
    has_rembg_model)
from nunif.device import mps_is_available, xpu_is_available
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS, has_nvenc
from nunif.utils.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_restore_all, persistent_manager_register,
    resolve_default_dir, extension_list_to_wildcard, validate_number,
    set_icon_ex, start_file, load_icon)
from .locales import LOCALES
from . import models # noqa
from .depth_anything_model import DepthAnythingModel
from .depth_pro_model import DepthProModel
from .depth_pro_model import MODEL_FILES as DEPTH_PRO_MODELS
from . import export_config
import torch


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
YAML_EXTENSIONS = extension_list_to_wildcard((".yml", ".yaml"))
CONFIG_PATH = path.join(path.dirname(__file__), "..", "tmp", "iw3-gui.cfg")
os.makedirs(path.dirname(CONFIG_PATH), exist_ok=True)


LAYOUT_DEBUG = False


LEVEL_LIBX264 = ["3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0", "5.1", "5.2", "6.0", "6.2"]
LEVEL_LIBX265 = ["3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "5.2", "6.0", "6.1", "6.2", "8.5"]
LEVEL_ALL = ["auto"] + sorted(list(set(LEVEL_LIBX264) | set(LEVEL_LIBX265)), key=lambda v: float(v))

TUNE_LIBX264 = ["film", "animation", "grain", "stillimage", "psnr"]
TUNE_LIBX265 = ["grain", "animation", "psnr", "fastdecode", "zerolatency"]
TUNE_NVENC = ["hq", "ll", "ull"]
TUNE_ALL = [""] + sorted(list(set(TUNE_LIBX264) | set(TUNE_LIBX265)))

PRESET_LIBX264 = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                  "medium", "slow", "slower", "veryslow", "placebo"]
PRESET_NVENC = ["fast", "medium", "slow"]
PRESET_ALL = PRESET_LIBX264


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
        self.suspend_event = threading.Event()
        self.suspend_pos = 0
        self.suspend_event.set()
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

        self.chk_exif_transpose = wx.CheckBox(self.pnl_file, label=T("EXIF Transpose"),
                                              name="chk_exif_transpose")
        self.chk_exif_transpose.SetValue(True)
        self.chk_exif_transpose.SetToolTip(T("Transpose images according to EXIF Orientaion Tag"))

        self.chk_metadata = wx.CheckBox(self.pnl_file, label=T("Add metadata to filename"),
                                        name="chk_metadata")
        self.chk_metadata.SetValue(False)

        self.sep_image_format = wx.StaticLine(self.pnl_file, size=(2, 16), style=wx.LI_VERTICAL)
        self.lbl_image_format = wx.StaticText(self.pnl_file, label=" " + T("Image Format"))
        self.cbo_image_format = wx.ComboBox(self.pnl_file, choices=["png", "jpeg", "webp"],
                                            style=wx.CB_READONLY, name="cbo_image_format")
        self.cbo_image_format.SetSelection(0)
        self.cbo_image_format.SetToolTip(T("Output Image Format"))

        sublayout = wx.BoxSizer(wx.HORIZONTAL)
        sublayout.Add(self.chk_resume, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.chk_recursive, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.chk_exif_transpose, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.chk_metadata, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.sep_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.lbl_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sublayout.Add(self.cbo_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

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
        self.cbo_method = wx.ComboBox(self.grp_stereo, choices=["row_flow_v3", "row_flow_v2", "forward_fill"],
                                      style=wx.CB_READONLY, name="cbo_method")
        self.cbo_method.SetSelection(0)

        self.lbl_stereo_width = wx.StaticText(self.grp_stereo, label=T("Stereo Procesing Width"))
        self.cbo_stereo_width = EditableComboBox(self.grp_stereo,
                                                 choices=["Default", "1920", "1280", "640"],
                                                 name="cbo_stereo_width")
        self.cbo_stereo_width.SetSelection(0)
        self.cbo_stereo_width.SetToolTip(T("Only used for row_flow_v3 and row_flow_v2"))

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        depth_models = [
            "ZoeD_N", "ZoeD_K", "ZoeD_NK",
            "ZoeD_Any_N", "ZoeD_Any_K",
            "DepthPro_SD", "DepthPro_HD", "DepthPro",
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

        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
                                           choices=depth_models,
                                           style=wx.CB_READONLY, name="cbo_depth_model")
        self.cbo_depth_model.SetSelection(0)

        self.lbl_zoed_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_zoed_resolution = EditableComboBox(self.grp_stereo,
                                                    choices=["Default", "512"],
                                                    name="cbo_zoed_resolution")
        self.cbo_zoed_resolution.SetSelection(0)

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(self.grp_stereo,
                                                     choices=["-3", "-2", "-1", "0", "1", "2", "3"],
                                                     name="cbo_foreground_scale")
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
            choices=["Full SBS", "Half SBS",
                     "Full TB", "Half TB",
                     "VR90",
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

        layout = wx.FlexGridSizer(rows=11, cols=2, vgap=4, hgap=4)
        layout.Add(self.lbl_divergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, 1, wx.EXPAND)
        layout.Add(self.lbl_convergence, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence, 1, wx.EXPAND)
        layout.Add(self.lbl_ipd_offset, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_ipd_offset, 1, wx.EXPAND)
        layout.Add(self.lbl_method, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, 1, wx.EXPAND)
        layout.Add(self.lbl_stereo_width, 0, wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_width, 1, wx.EXPAND)
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
        self.cbo_video_format = wx.ComboBox(self.grp_video, choices=["mp4", "mkv", "avi"],
                                            style=wx.CB_READONLY, name="cbo_video_format")
        self.cbo_video_format.SetSelection(0)

        self.lbl_video_codec = wx.StaticText(self.grp_video, label=T("Video Codec"))
        self.cbo_video_codec = EditableComboBox(
            self.grp_video, choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc", "utvideo"],
            name="cbo_video_codec")
        self.cbo_video_codec.SetSelection(0)

        self.lbl_fps = wx.StaticText(self.grp_video, label=T("Max FPS"))
        self.cbo_fps = EditableComboBox(
            self.grp_video, choices=["1000", "60", "59.94", "30", "29.97", "24", "23.976", "15", "1", "0.25"],
            name="cbo_fps")
        self.cbo_fps.SetSelection(3)

        self.lbl_pix_fmt = wx.StaticText(self.grp_video, label=T("Pixel Format"))
        self.cbo_pix_fmt = wx.ComboBox(self.grp_video, choices=["yuv420p", "yuv444p", "rgb24"],
                                       style=wx.CB_READONLY, name="cbo_pix_fmt")
        self.cbo_pix_fmt.SetSelection(0)

        self.lbl_colorspace = wx.StaticText(self.grp_video, label=T("Colorspace"))
        self.cbo_colorspace = wx.ComboBox(
            self.grp_video,
            choices=["auto", "unspecified", "bt709", "bt709-pc", "bt709-tv", "bt601", "bt601-pc", "bt601-tv"],
            style=wx.CB_READONLY, name="cbo_colorspace")
        self.cbo_colorspace.SetSelection(1)

        self.lbl_crf = wx.StaticText(self.grp_video, label=T("CRF"))
        self.cbo_crf = EditableComboBox(self.grp_video, choices=[str(n) for n in range(16, 28)],
                                        name="cbo_crf")
        self.cbo_crf.SetSelection(4)

        self.lbl_profile_level = wx.StaticText(self.grp_video, label=T("Level"))
        self.cbo_profile_level = EditableComboBox(self.grp_video, choices=LEVEL_ALL, name="cbo_profile_level")
        self.cbo_profile_level.SetSelection(0)

        self.lbl_preset = wx.StaticText(self.grp_video, label=T("Preset"))
        self.cbo_preset = wx.ComboBox(
            self.grp_video, choices=PRESET_ALL,
            style=wx.CB_READONLY, name="cbo_preset")
        self.cbo_preset.SetSelection(0)

        self.lbl_tune = wx.StaticText(self.grp_video, label=T("Tune"))
        self.cbo_tune = wx.ComboBox(
            self.grp_video, choices=TUNE_ALL,
            style=wx.CB_READONLY, name="cbo_tune")
        self.cbo_tune.SetSelection(0)
        self.chk_tune_fastdecode = wx.CheckBox(self.grp_video, label=T("fastdecode"),
                                               name="chk_tune_fastdecode")
        self.chk_tune_fastdecode.SetValue(False)
        self.chk_tune_zerolatency = wx.CheckBox(self.grp_video, label=T("zerolatency"),
                                                name="chk_tune_zerolatency")
        self.chk_tune_zerolatency.SetValue(False)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_fps, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_fps, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_format, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_format, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_codec, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_codec, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pix_fmt, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pix_fmt, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_colorspace, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_colorspace, (4, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_crf, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_crf, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_profile_level, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_profile_level, (6, 1), flag=wx.EXPAND)

        layout.Add(self.lbl_preset, (7, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_preset, (7, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tune, (8, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tune, (8, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_fastdecode, (9, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_zerolatency, (10, 1), flag=wx.EXPAND)

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
        elif mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif xpu_is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                self.cbo_device.Append(device_name, i)

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
        self.chk_cuda_stream = wx.CheckBox(self.grp_processor, label=T("Stream"), name="chk_cuda_stream")
        self.chk_cuda_stream.SetToolTip(T("Use per-thread CUDA Stream (experimental: fast or slow or crash)"))
        self.chk_cuda_stream.SetValue(False)

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_zoed_batch_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_zoed_batch_size, (1, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (2, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (3, 2), flag=wx.EXPAND)
        layout.Add(self.chk_cuda_stream, (3, 3), flag=wx.EXPAND)

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
        self.btn_suspend = wx.Button(self.pnl_process, label=T("Suspend"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Cancel"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.prg_tqdm, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        layout.Add(self.btn_start, 0, wx.ALL, 4)
        layout.Add(self.btn_suspend, 0, wx.ALL, 4)
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

        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_stereo_format)
        self.cbo_video_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_video_format)
        self.cbo_video_codec.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_video_codec)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)
        self.btn_suspend.Bind(wx.EVT_BUTTON, self.on_click_btn_suspend)

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
        self.btn_suspend.Disable()

        editable_comboxes = [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_zoed_resolution,
            self.cbo_stereo_width,
            self.cbo_edge_dilation,
            self.cbo_ema_decay,
            self.cbo_fps,
            self.cbo_crf,
            self.cbo_profile_level,
            self.cbo_video_codec,
            self.cbo_foreground_scale,
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
        self.update_video_format()
        self.update_video_codec()

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
        is_export = self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}
        if is_export:
            self.chk_resume.Enable()
            self.chk_recursive.Disable()
        else:
            if is_yaml(input_path):
                try:
                    config = export_config.ExportConfig.load(input_path)
                    if config.type == export_config.IMAGE_TYPE:
                        self.chk_resume.Enable()
                        self.chk_recursive.Disable()
                    else:
                        self.chk_resume.Disable()
                        self.chk_recursive.Disable()
                except:  # noqa
                    self.chk_resume.Disable()
                    self.chk_recursive.Disable()
            elif path.isdir(input_path) or is_text(input_path):
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
        args = self.parse_args()
        video = is_video(input_path)

        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename)
        elif is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, args, video=video))

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
        if (DepthAnythingModel.supported(name) or DepthProModel.supported(name) or name.startswith("ZoeD_Any_")):
            self.chk_edge_dilation.SetValue(True)
            self.cbo_edge_dilation.Enable()
        else:
            self.chk_edge_dilation.SetValue(False)
            self.cbo_edge_dilation.Disable()
        if name in DEPTH_PRO_MODELS:
            self.cbo_zoed_resolution.Disable()
            self.chk_fp16.Disable()
        else:
            self.cbo_zoed_resolution.Enable()
            self.chk_fp16.Enable()

    def update_video_format(self):
        name = self.cbo_video_format.GetValue()
        if name == "avi":
            self.cbo_profile_level.Disable()
            self.cbo_crf.Disable()
            self.cbo_preset.Disable()
            self.cbo_tune.Disable()
            self.chk_tune_fastdecode.Disable()
            self.chk_tune_zerolatency.Disable()
        else:
            self.cbo_profile_level.Enable()
            self.cbo_crf.Enable()
            self.cbo_preset.Enable()
            self.cbo_tune.Enable()
            self.chk_tune_fastdecode.Enable()
            self.chk_tune_zerolatency.Enable()

        # codec
        if name == "avi":
            choices = ["utvideo"]
        else:
            choices = ["libx264", "libx265"]
            if has_nvenc():
                choices += ["h264_nvenc", "hevc_nvenc"]

        user_codec = self.cbo_video_codec.GetValue()
        if user_codec not in {"libx265", "libx264", "h264_nvenc", "hevc_nvenc", "utvideo"}:
            choices.append(user_codec)
        self.cbo_video_codec.SetItems(choices)
        if user_codec in choices:
            self.cbo_video_codec.SetSelection(choices.index(user_codec))
        else:
            self.cbo_video_codec.SetSelection(0)
        self.update_video_codec()

    def update_video_codec(self):
        container_format = self.cbo_video_format.GetValue()
        codec = self.cbo_video_codec.GetValue()

        # level
        user_level = self.cbo_profile_level.GetValue()
        if codec in {"libx264", "h264_nvenc"}:
            choices = ["auto"] + LEVEL_LIBX264
        elif codec in {"libx265", "hevc_nvenc"}:
            choices = ["auto"] + LEVEL_LIBX265
        else:
            choices = LEVEL_ALL

        self.cbo_profile_level.SetItems(choices)
        if user_level in choices:
            self.cbo_profile_level.SetSelection(choices.index(user_level))
        else:
            self.cbo_profile_level.SetSelection(0)

        # preset
        if container_format in {"mp4", "mkv"}:
            preset = self.cbo_preset.GetValue()
            if codec in {"libx265", "libx264"}:
                # preset
                choices = PRESET_LIBX264
                default_preset = "ultrafast"
            elif codec in {"h264_nvenc", "hevc_nvenc"}:
                choices = PRESET_NVENC
                default_preset = "medium"
            else:
                choices = PRESET_ALL
                default_preset = "ultrafast"

            self.cbo_preset.SetItems(choices)
            if preset in choices:
                self.cbo_preset.SetSelection(choices.index(preset))
            else:
                self.cbo_preset.SetSelection(choices.index(default_preset))
        # tune
        if container_format in {"mp4", "mkv"}:
            if codec == "libx265":
                # tune
                tune = []
                if self.chk_tune_zerolatency.IsChecked():
                    tune.append("zerolatency")
                if self.chk_tune_fastdecode.IsChecked():
                    tune.append("fastdecode")
                tune.append(self.cbo_tune.GetValue())

                choices = [""] + TUNE_LIBX265
                self.cbo_tune.SetItems(choices)
                if tune[0] in choices:
                    self.cbo_tune.SetSelection(choices.index(tune[0]))
                else:
                    self.cbo_tune.SetSelection(0)

                self.chk_tune_fastdecode.SetValue(False)
                self.chk_tune_fastdecode.Disable()
                self.chk_tune_zerolatency.SetValue(False)
                self.chk_tune_zerolatency.Disable()

            elif codec == "libx264":
                tune = []
                tune.append(self.cbo_tune.GetValue())
                if self.chk_tune_zerolatency.IsChecked():
                    tune.append("zerolatency")
                if self.chk_tune_fastdecode.IsChecked():
                    tune.append("fastdecode")

                choices = [""] + TUNE_LIBX264
                self.cbo_tune.SetItems(choices)
                if tune[0] in choices:
                    self.cbo_tune.SetSelection(choices.index(tune[0]))
                else:
                    self.cbo_tune.SetSelection(0)

                self.chk_tune_fastdecode.Enable()
                self.chk_tune_fastdecode.SetValue("fastdecode" in tune)
                self.chk_tune_zerolatency.Enable()
                self.chk_tune_zerolatency.SetValue("zerolatency" in tune)
            elif codec in {"h264_nvenc", "hevc_nvenc"}:
                tune = self.cbo_tune.GetValue()
                choices = [""] + TUNE_NVENC
                self.cbo_tune.SetItems(choices)
                if tune in choices:
                    self.cbo_tune.SetSelection(choices.index(tune))
                else:
                    self.cbo_tune.SetSelection(0)
                self.chk_tune_fastdecode.SetValue(False)
                self.chk_tune_fastdecode.Disable()
                self.chk_tune_zerolatency.SetValue(False)
                self.chk_tune_zerolatency.Disable()

    def on_selected_index_changed_cbo_depth_model(self, event):
        self.update_model_selection()

    def on_selected_index_changed_cbo_stereo_format(self, event):
        self.update_input_option_state()

    def on_selected_index_changed_cbo_video_format(self, event):
        self.update_video_format()

    def on_selected_index_changed_cbo_video_codec(self, event):
        self.update_video_codec()

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

    def confirm_overwrite(self, args):
        input_path = args.input
        output_path = args.output
        video = is_video(input_path)
        resume = args.resume
        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename, export_config.FILENAME)
            else:
                output_path = path.join(output_path, export_config.FILENAME)
        else:
            if is_output_dir(output_path):
                output_path = path.join(
                    output_path,
                    make_output_filename(input_path, args, video=video))
            else:
                output_path = output_path
                resume = False

        if path.exists(output_path) and not resume:
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

    def parse_args(self):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return None
        if not validate_number(self.cbo_pad.GetValue(), 0.0, 10.0, allow_empty=True):
            self.show_validation_error_message(T("Padding"), 0.0, 10.0)
            return None
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.cbo_fps.GetValue(), 0.25, 1000.0, allow_empty=False):
            self.show_validation_error_message(T("Max FPS"), 0.25, 1000.0)
            return None
        if not validate_number(self.cbo_crf.GetValue(), 0, 51, is_int=True):
            self.show_validation_error_message(T("CRF"), 0, 51)
            return None
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return None
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return None

        zoed_height = self.cbo_zoed_resolution.GetValue()
        if zoed_height == "Default" or zoed_height == "":
            zoed_height = None
        else:
            if not validate_number(zoed_height, 384, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 384, 8190)
                return
            zoed_height = int(zoed_height)

        stereo_width = self.cbo_stereo_width.GetValue()
        if stereo_width == "Default" or stereo_width == "":
            stereo_width = None
        else:
            if not validate_number(stereo_width, 320, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Stereo processing Width"), 320, 8190)
                return
            stereo_width = int(stereo_width)

        parser = create_parser(required_true=False)

        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        tb = self.cbo_stereo_format.GetValue() == "Full TB"
        half_tb = self.cbo_stereo_format.GetValue() == "Half TB"
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
        profile_level = self.cbo_profile_level.GetValue()
        if not profile_level or profile_level == "auto":
            profile_level = None

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
        resume = self.chk_resume.IsEnabled() and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None
        edge_dilation = int(self.cbo_edge_dilation.GetValue()) if self.chk_edge_dilation.IsChecked() else 0
        metadata = "filename" if self.chk_metadata.GetValue() else None

        parser.set_defaults(
            input=input_path,
            output=self.txt_output.GetValue(),
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            ipd_offset=float(self.sld_ipd_offset.GetValue()),
            method=self.cbo_method.GetValue(),
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            edge_dilation=edge_dilation,
            vr180=vr180,
            half_sbs=half_sbs,
            tb=tb,
            half_tb=half_tb,
            anaglyph=anaglyph,
            export=export,
            export_disparity=export_disparity,
            debug_depth=debug_depth,
            ema_normalize=self.chk_ema_normalize.GetValue(),
            ema_decay=float(self.cbo_ema_decay.GetValue()),

            max_fps=float(self.cbo_fps.GetValue()),
            pix_fmt=self.cbo_pix_fmt.GetValue(),
            colorspace=self.cbo_colorspace.GetValue(),
            video_format=self.cbo_video_format.GetValue(),
            format=self.cbo_image_format.GetValue(),
            video_codec=self.cbo_video_codec.GetValue(),
            crf=int(self.cbo_crf.GetValue()),
            profile_level=profile_level,
            preset=self.cbo_preset.GetValue(),
            tune=list(tune),

            remove_bg=remove_bg,
            bg_model=bg_model_type,

            pad=pad,
            rotate_right=rotate_right,
            rotate_left=rotate_left,
            disable_exif_transpose=not self.chk_exif_transpose.GetValue(),
            vf=vf,
            max_output_width=max_output_width,
            max_output_height=max_output_height,
            keep_aspect_ratio=self.chk_keep_aspect_ratio.GetValue(),

            gpu=device_id,
            zoed_batch_size=int(self.cbo_zoed_batch_size.GetValue()),
            zoed_height=zoed_height,
            stereo_width=stereo_width,
            max_workers=int(self.cbo_max_workers.GetValue()),
            tta=self.chk_tta.GetValue(),
            disable_amp=not self.chk_fp16.GetValue(),
            low_vram=self.chk_low_vram.GetValue(),
            cuda_stream=self.chk_cuda_stream.GetValue(),

            resume=resume,
            recursive=recursive,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            suspend_event=self.suspend_event,
            tqdm_fn=functools.partial(TQDMGUI, self),
            depth_model=self.depth_model)
        return args

    def on_click_btn_start(self, event):
        args = self.parse_args()
        if args is None:
            return
        if not self.confirm_overwrite(args):
            return

        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.btn_suspend.Enable()
        self.stop_event.clear()
        self.suspend_event.set()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        if args.state["depth_model"].has_checkpoint_file(args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {args.depth_model}...")
            if args.remove_bg and not has_rembg_model(args.bg_model):
                self.SetStatusText(f"Downloading {args.bg_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {args.depth_model}...")

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
        self.btn_suspend.Disable()
        self.btn_suspend.SetLabel(T("Suspend"))
        self.update_start_button_state()

        # free vram
        gc.collect()
        if torch.cuda.is_available:
            torch.cuda.empty_cache()

    def on_click_btn_cancel(self, event):
        self.suspend_event.set()
        self.stop_event.set()

    def on_click_btn_suspend(self, event):
        if self.suspend_event.is_set():
            self.suspend_event.clear()
            self.btn_suspend.SetLabel(T("Resume"))
        else:
            self.start_time = time()
            self.suspend_pos = self.prg_tqdm.GetValue()
            self.suspend_event.set()
            self.btn_suspend.SetLabel(T("Suspend"))

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            if 0 < value:
                self.prg_tqdm.SetRange(value)
            else:
                self.prg_tqdm.SetRange(1)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.suspend_pos = 0
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
            fps = (pos - self.suspend_pos) / (now - self.start_time + 1e-6)
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
