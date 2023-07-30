import wx
from wx.lib.masked.timectrl import TimeCtrl as _TimeCtrl
import os
from os import path
import sys
import subprocess


myEVT_TQDM = wx.NewEventType()
EVT_TQDM = wx.PyEventBinder(myEVT_TQDM, 1)


class TQDMEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, type=None, value=None, desc=None):
        super(TQDMEvent, self).__init__(etype, eid)
        self.type = type
        self.value = value
        self.desc = desc

    def GetValue(self):
        return (self.type, self.value, self.desc)


class TQDMGUI():
    def __init__(self, parent, **kwargs):
        self.parent = parent
        total = kwargs["total"]
        self.desc = kwargs.get("desc", "")
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 0, total, self.desc))

    def update(self, n=1):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 1, n, self.desc))

    def close(self):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 2, 0, self.desc))


class FileDropCallback(wx.FileDropTarget):
    def __init__(self, callback):
        super(FileDropCallback, self).__init__()
        self.callback = callback

    def OnDropFiles(self, x, y, filenames):
        return self.callback(x, y, filenames)


# Fix TimeCtrl
# ref: https://github.com/wxWidgets/Phoenix/issues/639#issuecomment-356129566
class TimeCtrl(_TimeCtrl):
    def __init__(self, parent, **kwargs):
        super(TimeCtrl, self).__init__(parent, **kwargs)
        if sys.platform != "win32":
            self.Unbind(wx.EVT_CHAR)
            self.Bind(wx.EVT_CHAR_HOOK, self._OnChar)


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


def resolve_default_dir(src):
    if src:
        if path.isfile(src):
            default_dir = path.dirname(src)
        elif path.isdir(src):
            default_dir = src
        elif "." in path.basename(src):
            default_dir = path.dirname(src)
        else:
            default_dir = src
    else:
        default_dir = ""
    return default_dir


def extension_list_to_wildcard(extensions):
    extensions = list(extensions)
    if sys.platform != "win32":
        # wx.FileDialog does not find uppercase extensions on Linux so add them
        extensions = extensions + [ext.upper() for ext in extensions]
    return ";".join(["*" + ext for ext in extensions])


def set_icon_ex(main_frame, icon_path, app_id):
    icons = wx.IconBundle(icon_path)
    main_frame.SetIcons(icons)
    if sys.platform == "win32":
        # Set AppUserModelID to show correct icon on Taskbar
        try:
            from ctypes import windll
            from win32com.propsys import propsys, pscon
            import pythoncom
            windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            hwnd = main_frame.GetHandle()
            propStore = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
            propStore.SetValue(pscon.PKEY_AppUserModel_ID,
                               propsys.PROPVARIANTType(app_id, pythoncom.VT_ILLEGAL))
            propStore.Commit()
        except: # noqa
            pass


def start_file(file_path):
    if not path.exists(file_path):
        return

    fd = subprocess.DEVNULL
    options = {"stderr": fd, "stdout": fd, "stdin": fd}
    if sys.platform == "win32":
        if path.isdir(file_path):
            subprocess.Popen(["explorer", file_path], shell=True, **options)
        else:
            subprocess.Popen(["start", file_path], shell=True, **options)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", file_path], start_new_session=True, **options)
    elif sys.platform == "darwin":
        # Not tested
        subprocess.Popen(["open", file_path], **options)
    else:
        print("start_file: unknown platform", file=sys.stderr)


def load_icon(name):
    return wx.Bitmap(path.join(path.dirname(__file__), "..", "rc", "icons", name))
