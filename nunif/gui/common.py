import wx
from wx.lib.buttons import GenBitmapButton
from wx.lib.masked.timectrl import TimeCtrl as _TimeCtrl
from wx.lib.masked.ipaddrctrl import IpAddrCtrl as _IpAddrCtrl
import wx.lib.agw.persist as persist
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
        self.SetDefaultAction(wx.DragCopy)

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


class IpAddrCtrl(_IpAddrCtrl):
    def __init__(self, parent, **kwargs):
        super(IpAddrCtrl, self).__init__(parent, **kwargs)
        if sys.platform != "win32":
            self.Unbind(wx.EVT_CHAR)
            self.Bind(wx.EVT_CHAR_HOOK, self._OnChar)


class EditableComboBox(wx.ComboBox):
    """
    Serializable Editable ComboBox
    wx.ComboBox can not serialize Value
    """
    def __init__(self, parent, **kwargs):
        if "style" in kwargs:
            style = kwargs.get("style", 0) | wx.CB_DROPDOWN
            kwargs.pop("style")
        else:
            style = wx.CB_DROPDOWN
        super().__init__(parent, style=style, **kwargs)


class EditableComboBoxPersistentHandler(persist.AbstractHandler):
    def Save(self):
        combo, obj = self._window, self._pObject
        value = combo.GetValue()
        obj.SaveCtrlValue("Value", value)
        return True

    def Restore(self):
        combo, obj = self._window, self._pObject
        value = obj.RestoreCtrlValue("Value")
        if value is not None:
            if value in combo.GetStrings():
                combo.SetStringSelection(value)
            else:
                combo.SetValue(value)
            return True
        return False

    def GetKind(self):
        return "nunif.EditableComboBox"


def persistent_manager_register_all(manager, window):
    # register all child controls without Restore() call
    if window.GetName() not in persist.BAD_DEFAULT_NAMES and persist.HasCtrlHandler(window):
        manager.Register(window)

    for child in window.GetChildren():
        persistent_manager_register_all(manager, child)


def persistent_manager_restore_all(manager, exclude_names={}):
    # restore all registered controls
    for name, obj in list(manager._persistentObjects.items()):  # NOTE: private attribute
        if name not in exclude_names:
            manager.Restore(obj.GetWindow())


def persistent_manager_unregister_all(manager):
    for name, obj in list(manager._persistentObjects.items()):  # NOTE: private attribute
        manager.Unregister(obj.GetWindow())


def persistent_manager_register(manager, window, handler):
    # override
    manager.Unregister(window)
    manager.Register(window, handler)


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
    if file_path.startswith("http://") or file_path.startswith("https://"):
        pass
    elif not path.exists(file_path):
        return

    fd = subprocess.DEVNULL
    options = {"stderr": fd, "stdout": fd, "stdin": fd}
    if sys.platform == "win32":
        if path.isdir(file_path):
            subprocess.Popen(["explorer", file_path], shell=True, **options)
        else:
            subprocess.Popen(["start", "", file_path], shell=True, **options)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", file_path], start_new_session=True, **options)
    elif sys.platform == "darwin":
        # Not tested
        subprocess.Popen(["open", file_path], **options)
    else:
        print("start_file: unknown platform", file=sys.stderr)


def load_icon(name):
    return wx.Bitmap(path.join(path.dirname(__file__), "..", "rc", "icons", name))


def apply_dark_mode(
        window,
        fg_color=wx.Colour(*(0xdc,) * 3),
        bg_color=wx.Colour(*(0x1e,) * 3),
        btn_color=wx.Colour(*(0x33,) * 3)
):
    if isinstance(window, wx.StaticLine):
        window.SetBackgroundColour(fg_color)
    elif isinstance(window, (wx.Button, wx.StatusBar, GenBitmapButton)):
        window.SetForegroundColour(fg_color)
        window.SetBackgroundColour(btn_color)
    else:
        window.SetForegroundColour(fg_color)
        window.SetBackgroundColour(bg_color)

    window.Refresh()

    for child in window.GetChildren():
        apply_dark_mode(child, fg_color, bg_color, btn_color)


def is_dark_mode():
    if sys.platform != "win32":
        return False
    else:
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return value == 0
        except: # noqa
            return False
