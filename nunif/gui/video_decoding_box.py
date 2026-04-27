import wx
from nunif.utils.video import HW_DEVICES


def empty_translate_function(s):
    return s


class VideoDecodingBox():
    def __init__(
            self,
            parent,
            name_prefix="",
            translate_function=empty_translate_function,
            **kwargs
    ):
        T = translate_function
        prefix = name_prefix + "_" if name_prefix else ""
        self.grp_video_dec = wx.StaticBox(parent, label=T("Video Decoding"), **kwargs)

        self.lbl_hwaccel = wx.StaticText(self.grp_video_dec, label=T("HWAccel"))
        self.cbo_hwaccel = wx.ComboBox(self.grp_video_dec, choices=[""] + HW_DEVICES,
                                       name=f"{prefix}cbo_hwaccel")
        self.cbo_hwaccel.SetEditable(False)
        self.cbo_hwaccel.SetSelection(0)
        self.chk_software_fallback = wx.CheckBox(self.grp_video_dec, label=T("Software Fallback"),
                                                 name=f"{prefix}chk_software_fallback")
        self.chk_software_fallback.SetValue(True)
        self.chk_software_fallback.SetToolTip(T("Use software decoder if hardware acceleration fails or is unsupported"))

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.lbl_hwaccel, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_hwaccel, (0, 1), flag=wx.EXPAND)
        layout.Add(self.chk_software_fallback, (1, 1), flag=wx.EXPAND)

        self.box_sizer = wx.StaticBoxSizer(self.grp_video_dec, wx.VERTICAL)
        self.box_sizer.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

    def get_editable_comboboxes(self):
        return []

    @property
    def hwaccel(self):
        val = self.cbo_hwaccel.GetValue()
        if not val:
            return None
        return val

    @property
    def software_fallback(self):
        return self.chk_software_fallback.IsChecked()

    @property
    def sizer(self):
        return self.box_sizer
