import nunif.pythonw_fix  # noqa
import locale
import sys
import os
from os import path
import gc
import functools
from time import time
import threading
from pprint import pprint # noqa
import wx
from wx.lib.masked.numctrl import NumCtrl # noqa
from wx.lib.buttons import GenBitmapButton
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as wxpm
from .ui_utils import (
    create_parser, set_state_args, waifu2x_main,
    is_video, is_output_dir, is_text, is_image,
    MODEL_DIR, DEFAULT_ART_MODEL_DIR,
    DEFAULT_ART_SCAN_MODEL_DIR, DEFAULT_PHOTO_MODEL_DIR)
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS
from nunif.utils.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    resolve_default_dir, extension_list_to_wildcard,
    set_icon_ex, start_file, load_icon)
from .locales import LOCALES
from . import models # noqa
import torch


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
CONFIG_PATH = path.join(path.dirname(__file__), "..", "tmp", "waifu2x-gui.cfg")
os.makedirs(path.dirname(CONFIG_PATH), exist_ok=True)


LAYOUT_DEBUG = False


class Waifu2xApp(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker(main_frame.GetTitle())
        if self.instance.IsAnotherRunning():
            wx.MessageBox(T("Another instance is running"), T("Error"), style=wx.ICON_ERROR)
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
            name="waifu2x-gui",
            title=T("waifu2x-gui"),
            size=(1000, 740),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
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

        # Superresolution settings
        # NOTE: term is translated with webgen locale. See WEBGEN_TERMS in ./locales.py
        self.grp_sr = wx.StaticBox(self.pnl_options, label=T("Superresolution"))
        self.opt_model = wx.RadioBox(
            self.grp_sr, label=T("Model"),
            choices=[T("artwork"), T("artwork") + "/" + T("scan"),
                     T("photo"), "cunet/art", "upconv_7/art", "upconv_7/photo"],
            majorDimension=3, name="opt_model")
        self.model_dirs = [
            DEFAULT_ART_MODEL_DIR, DEFAULT_ART_SCAN_MODEL_DIR, DEFAULT_PHOTO_MODEL_DIR,
            path.join(MODEL_DIR, "cunet", "art"),
            path.join(MODEL_DIR, "upconv_7", "art"),
            path.join(MODEL_DIR, "upconv_7", "photo"),
        ]
        self.model_tta_support = [True, False, False, True, True, True]
        self.model_4x_support = [True, True, True, False, False, False]

        self.opt_model.SetSelection(0)
        self.opt_model.SetItemToolTip(0, T("Anime Style Art, Cliparts"))
        self.opt_model.SetItemToolTip(1, T("Manga, Anime Screencaps, Anime Style Art for more clear results"))
        self.opt_model.SetItemToolTip(2, T("Photograph"))
        self.opt_model.SetItemToolTip(3, T("Old version, Art model, fast"))
        self.opt_model.SetItemToolTip(4, T("Old version, Art model, veryfast"))
        self.opt_model.SetItemToolTip(5, T("Old version, Photo model, veryfast"))

        self.opt_noise_level = wx.RadioBox(
            self.grp_sr, label=T("noise_reduction"),
            choices=[T("nr_none"), T("nr_low"), T("nr_medium"), T("nr_high"), T("nr_highest")],
            name="opt_noise_level")
        self.opt_noise_level.SetSelection(1)

        self.opt_upscaling = wx.RadioBox(
            self.grp_sr, label=T("upscaling"),
            choices=[T("up_none"), "2x", "4x"],
            name="opt_upscaling")
        self.opt_upscaling.SetSelection(1)

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.opt_model, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.opt_upscaling, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.opt_noise_level, 0, wx.ALL | wx.EXPAND, border=4)
        sizer_sr = wx.StaticBoxSizer(self.grp_sr, wx.VERTICAL)
        sizer_sr.Add(layout, 1, wx.ALL | wx.EXPAND, 8)

        # video encoding
        # max-fps, crf, preset, tune
        self.grp_video = wx.StaticBox(self.pnl_options, label=T("Video Encoding"))

        self.lbl_fps = wx.StaticText(self.grp_video, label=T("Max FPS"))
        self.cbo_fps = wx.ComboBox(self.grp_video, choices=["0.25", "1", "15", "30", "60", "1000"],
                                   style=wx.CB_READONLY, name="cbo_fps")
        self.cbo_fps.SetSelection(3)

        self.lbl_pix_fmt = wx.StaticText(self.grp_video, label=T("Pixel Format"))
        self.cbo_pix_fmt = wx.ComboBox(self.grp_video, choices=["yuv420p", "yuv444p"],
                                       style=wx.CB_READONLY, name="cbo_pix_fmt")
        self.cbo_pix_fmt.SetSelection(0)

        self.lbl_crf = wx.StaticText(self.grp_video, label=T("CRF"))
        self.cbo_crf = wx.ComboBox(self.grp_video, choices=[str(n) for n in range(16, 28 + 1)],
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
        layout.Add(self.lbl_pix_fmt, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pix_fmt, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_crf, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_crf, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_preset, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_preset, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tune, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tune, (4, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_fastdecode, (5, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_zerolatency, (6, 1), flag=wx.EXPAND)

        sizer_video = wx.StaticBoxSizer(self.grp_video, wx.VERTICAL)
        sizer_video.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

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

        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=(200, -1),
                                      style=wx.CB_READONLY, name="cbo_rotate")
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")
        """
        self.chk_grain_noise = wx.CheckBox(self.grp_video_filter,
                                           label=T("Add grain noise"), name="chk_grain_noise")
        self.txt_grain_noise = NumCtrl(self.grp_video_filter, name="txt_grain_noise")
        self.txt_grain_noise.SetFractionWidth(2)
        self.txt_grain_noise.SetMax(0.5)
        self.txt_grain_noise.SetMin(0.0)
        self.txt_grain_noise.SetValue(0.05)
        """

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_start_time, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (0, 1), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_deinterlace, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (4, 1), flag=wx.EXPAND)
        # layout.Add(self.chk_grain_noise, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        # layout.Add(self.txt_grain_noise, (5, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=(240, -1), style=wx.CB_READONLY,
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

        self.lbl_tile_size = wx.StaticText(self.grp_processor, label=T("Tile Size"))
        self.cbo_tile_size = wx.ComboBox(self.grp_processor,
                                         choices=["1024", "640", "400", "256", "64"],
                                         style=wx.CB_READONLY, name="cbo_tile_size")
        self.cbo_tile_size.SetSelection(3)
        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor,
                                          choices=["64", "32", "16", "8", "4", "2", "1"],
                                          style=wx.CB_READONLY, name="cbo_batch_size")
        self.cbo_batch_size.SetSelection(4)

        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(T("Use flip augmentation to improve quality (veryslow)") + "\n" +
                                T("Ignored in some models"))
        self.chk_amp = wx.CheckBox(self.grp_processor, label=T("FP16 (fast)"), name="chk_amp")
        self.chk_amp.SetValue(True)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tile_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tile_size, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (2, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_amp, (3, 1), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        layout = wx.GridBagSizer(wx.HORIZONTAL)
        layout.Add(sizer_sr, (0, 0), (0, 4), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, (1, 0), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video_filter, (1, 2), flag=wx.ALL | wx.EXPAND, border=4)
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

        self.opt_model.Bind(wx.EVT_RADIOBOX, self.on_selected_index_changed_opt_model)
        self.opt_upscaling.Bind(wx.EVT_RADIOBOX, self.on_selected_index_changed_opt_upscaling)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.SetDropTarget(FileDropCallback(self.on_drop_files))
        # Disable default drop target
        for control in (self.txt_input, self.txt_output, self.txt_vf):
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
        self.update_upscaling_state()
        self.update_noise_level_state()
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

    def update_upscaling_state(self):
        if self.model_4x_support[self.opt_model.GetSelection()]:
            self.opt_upscaling.EnableItem(2, True)
        else:
            if self.opt_upscaling.GetSelection() == 2:
                self.opt_upscaling.SetSelection(1)
            self.opt_upscaling.EnableItem(2, False)

    def update_noise_level_state(self):
        if self.opt_upscaling.GetSelection() == 0:
            # 1x
            if self.opt_noise_level.GetSelection() == 0:
                self.opt_noise_level.SetSelection(1)
            self.opt_noise_level.EnableItem(0, False)
        else:
            self.opt_noise_level.EnableItem(0, True)

    def update_start_button_state(self):
        if not self.processing:
            if self.txt_input.GetValue() and self.txt_output.GetValue():
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()

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

    def on_selected_index_changed_opt_model(self, event):
        self.update_upscaling_state()

    def on_selected_index_changed_opt_upscaling(self, event):
        self.update_noise_level_state()

    def set_same_output_dir(self):
        selected_path = self.txt_input.GetValue()
        if path.isdir(selected_path):
            self.txt_output.SetValue(path.join(selected_path, "waifu2x"))
        else:
            self.txt_output.SetValue(path.join(path.dirname(selected_path), "waifu2x"))

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

        if is_output_dir(output_path):
            if is_video(input_path):
                output_path = path.join(
                    output_path,
                    path.splitext(path.basename(input_path))[0] + ".mp4")
            elif is_image(input_path):
                output_path = path.join(
                    output_path,
                    path.splitext(path.basename(input_path))[0] + ".png")

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
        self.update_input_option_state()
        self.reset_time_range()

    def on_text_changed_txt_output(self, event):
        self.update_start_button_state()

    def confirm_overwrite(self):
        input_path = self.txt_input.GetValue()
        output_path = self.txt_output.GetValue()

        if is_output_dir(output_path):
            output_path = path.join(output_path, path.basename(input_path))
        else:
            output_path = output_path

        if path.exists(output_path) and is_video(output_path):
            with wx.MessageDialog(None,
                                  message=output_path + "\n" + T("already exists. Overwrite?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                return dlg.ShowModal() == wx.ID_YES
        else:
            return True

    def on_click_btn_start(self, event):
        if not self.confirm_overwrite():
            return
        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.stop_event.clear()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        parser = create_parser(required_true=False)

        tune = set()
        if self.chk_tune_zerolatency.GetValue():
            tune.add("zerolatency")
        if self.chk_tune_fastdecode.GetValue():
            tune.add("fastdecode")
        if self.cbo_tune.GetValue():
            tune.add(self.cbo_tune.GetValue())
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

        device = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device == -2:
            # All CUDA
            gpus = list(range(torch.cuda.device_count()))
        else:
            gpus = [device]

        noise_level = int(self.opt_noise_level.GetSelection()) - 1
        scale = 2 ** (int(self.opt_upscaling.GetSelection()))
        assert noise_level in {-1, 0, 1, 2, 3}
        assert scale in {1, 2, 4}
        if scale == 1:
            method = "noise"
        else:
            if noise_level >= 0:
                method = f"noise_scale{scale}x"
            else:
                method = f"scale{scale}x"

        input_path = self.txt_input.GetValue()
        resume = (path.isdir(input_path) or is_text(input_path)) and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None
        tta = self.chk_tta.GetValue() and self.model_tta_support[self.opt_model.GetSelection()]

        parser.set_defaults(
            input=input_path,
            output=self.txt_output.GetValue(),
            model_dir=self.model_dirs[self.opt_model.GetSelection()],
            noise_level=noise_level,
            method=method,
            yes=True,  # TODO: remove this
            max_fps=float(self.cbo_fps.GetValue()),
            crf=int(self.cbo_crf.GetValue()),
            preset=self.cbo_preset.GetValue(),
            tune=list(tune),

            rotate_right=rotate_right,
            rotate_left=rotate_left,
            vf=vf,
            # TODO
            # grain=(self.txt_grain_noise.GetValue() > 0 and self.chk_grain_noise.GetValue()),
            # grain_strength=self.txt_grain_noise.GetValue(),

            gpu=gpus,
            batch_size=int(self.cbo_batch_size.GetValue()),
            tile_size=int(self.cbo_tile_size.GetValue()),
            tta=tta,
            disable_amp=not self.chk_amp.GetValue(),

            resume=resume,
            recursive=recursive,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            tqdm_fn=functools.partial(TQDMGUI, self))
        startWorker(self.on_exit_worker, waifu2x_main, wargs=(args,))
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
            e_type, e, stacktrace = sys.exc_info()
            message = getattr(e, "message", str(e))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.update_start_button_state()

        # free vram
        gc.collect()
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
            self.SetStatusText(f"0/{value} {desc}")
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
LOCALE_DICT_EN = LOCALES["en_US"]


def T(s):
    if s in LOCALE_DICT:
        return LOCALE_DICT[s]
    if s in LOCALE_DICT_EN:
        return LOCALE_DICT_EN[s]
    return s


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

    app = Waifu2xApp()
    app.MainLoop()


if __name__ == "__main__":
    main()
