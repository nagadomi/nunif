import wx
from os import path
import platform


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


class FileDropCallback(wx.FileDropTarget):
    def __init__(self, callback):
        super(FileDropCallback, self).__init__()
        self.callback = callback

    def OnDropFiles(self, x, y, filenames):
        return self.callback(x, y, filenames)


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
    if platform.system() != "Windows":
        # wx.FileDialog does not find uppercase extensions on Linux so add them
        extensions = extensions + [ext.upper() for ext in extensions]
    return ";".join(["*" + ext for ext in extensions])
