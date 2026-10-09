import nunif.pythonw_fix  # noqa
import os
import sys
import threading
import traceback
from os import path
import time
import wx
from wx.lib.delayedresult import startWorker
from wx.lib.buttons import GenBitmapButton
import wx.lib.agw.persist as persist
from wx.lib.intctrl import IntCtrl
from nunif.gui import (
    IpAddrCtrl,
    validate_number,
    set_icon_ex,
    load_icon,
    start_file,
    is_dark_mode,
    apply_dark_mode,
    persistent_manager_register_all,
    persistent_manager_unregister_all,
    persistent_manager_restore_all,
    get_default_locale,
)
from iw3.locales import LOCALES, load_language_setting
from .server import (
    server_main as backend_main,
    create_parser,
    set_state_args,
    get_local_address,
    is_private_address,
    is_loopback_address,
)
from .dir_config import TMP_DIR, EXAMPLE_DIR


# Configuration for persistence
CONFIG_DIR = TMP_DIR
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-player-gui.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-player-lang.cfg")
DEFAULT_ROOT_DIR = EXAMPLE_DIR
os.makedirs(CONFIG_DIR, exist_ok=True)


class IW3PlayerApp(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker("iw3-player.lock", CONFIG_DIR)
        if self.instance.IsAnotherRunning():
            wx.MessageBox("Another instance is running", "Error", wx.OK | wx.ICON_ERROR)
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
            name="iw3-player",
            title="iw3-player",
            size=(460, 380),
            style=wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX,
        )
        self.stop_event = threading.Event()
        self.processing = False
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

    def initialize_component(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # 1. Root Directory Selection
        grp_root = wx.StaticBox(panel, label=T("Media Directory"))
        root_sizer = wx.StaticBoxSizer(grp_root, wx.HORIZONTAL)

        self.txt_root = wx.TextCtrl(grp_root, name="txt_root")
        self.txt_root.SetValue(DEFAULT_ROOT_DIR)
        self.btn_browse_root = GenBitmapButton(grp_root, bitmap=load_icon("folder-open.png"))
        self.btn_browse_root.SetToolTip(T("Choose a directory"))

        root_sizer.Add(self.txt_root, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        root_sizer.Add(self.btn_browse_root, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        main_sizer.Add(root_sizer, 0, wx.ALL | wx.EXPAND, 10)

        # 2. Network Settings
        grp_network = wx.StaticBox(panel, label=T("Network"))
        network_grid = wx.GridBagSizer(vgap=5, hgap=10)

        self.chk_bind_addr = wx.CheckBox(grp_network, label=T("Address"), name="chk_bind_addr")
        self.txt_bind_addr = IpAddrCtrl(grp_network, size=self.FromDIP((200, -1)), name="txt_bind_addr")
        self.chk_bind_addr.SetValue(False)
        self.txt_bind_addr.SetValue("127.0.0.1")

        self.lbl_bind_addr_warning = wx.StaticText(grp_network, label="")
        self.lbl_bind_addr_warning.SetForegroundColour(wx.Colour(200, 0, 0))
        self.lbl_bind_addr_warning.Hide()

        self.btn_detect_ip = GenBitmapButton(grp_network, bitmap=load_icon("view-refresh.png"))
        self.btn_detect_ip.SetToolTip(T("Detect"))

        self.lbl_port = wx.StaticText(grp_network, label=T("Port"))
        self.txt_port = IntCtrl(
            grp_network, size=self.FromDIP((200, -1)), allow_none=False, min=1025, max=65535, name="txt_port"
        )
        self.txt_port.SetValue(1304)

        self.sep_network1 = wx.StaticLine(grp_network, style=wx.LI_HORIZONTAL)
        self.chk_auth = wx.CheckBox(grp_network, label=T("Basic Authentication"), name="chk_auth_enable")

        self.lbl_user = wx.StaticText(grp_network, label=T("Username"))
        self.txt_user = wx.TextCtrl(grp_network, name="txt_user")

        self.lbl_password = wx.StaticText(grp_network, label=T("Password"))
        self.txt_password = wx.TextCtrl(grp_network, style=wx.TE_PASSWORD, name="txt_password")

        network_grid.Add(self.chk_bind_addr, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        network_grid.Add(self.txt_bind_addr, (0, 1), flag=wx.EXPAND)
        network_grid.Add(self.btn_detect_ip, (0, 2))
        network_grid.Add(self.lbl_bind_addr_warning, (1, 0), (1, 3), flag=wx.EXPAND)
        network_grid.Add(self.lbl_port, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        network_grid.Add(self.txt_port, (2, 1), flag=wx.EXPAND)
        network_grid.Add(self.sep_network1, (3, 0), (1, 3), flag=wx.EXPAND | wx.ALL)
        network_grid.Add(self.chk_auth, (4, 0), (1, 3))
        network_grid.Add(self.lbl_user, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        network_grid.Add(self.txt_user, (5, 1), flag=wx.EXPAND)
        network_grid.Add(self.lbl_password, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        network_grid.Add(self.txt_password, (6, 1), flag=wx.EXPAND)
        network_grid.SetEmptyCellSize((0, 0))

        network_sizer = wx.StaticBoxSizer(grp_network, wx.VERTICAL)
        network_sizer.Add(network_grid, 0, wx.ALL | wx.EXPAND, 5)
        main_sizer.Add(network_sizer, 0, wx.ALL | wx.EXPAND, 10)

        # 4. Process Controls
        process_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_start = wx.Button(panel, label=T("Start"))
        self.btn_stop = wx.Button(panel, label=T("Shutdown"))
        self.btn_stop.Disable()

        self.txt_url = wx.TextCtrl(panel, size=self.FromDIP((200, -1)), style=wx.TE_READONLY)
        self.btn_url = GenBitmapButton(panel, bitmap=load_icon("go-next.png"))
        self.btn_url.Disable()

        process_sizer.Add((4, 0), 0, wx.ALL, 2)
        process_sizer.Add(self.btn_start, 0, wx.ALL, 2)
        process_sizer.Add(self.btn_stop, 0, wx.ALL, 2)
        process_sizer.Add(self.txt_url, 0, wx.ALL | wx.EXPAND, 2)
        process_sizer.Add(self.btn_url, 0, wx.ALL, 2)

        main_sizer.Add(process_sizer, 0, wx.CENTER)
        panel.SetSizer(main_sizer)

        layout = wx.GridBagSizer(vgap=0, hgap=0)
        layout.Add(panel, (0, 0), flag=wx.ALL | wx.EXPAND, border=0)
        layout.SetEmptyCellSize((0, 0))
        self.SetSizer(layout)

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(panel.GetBackgroundColour())

        # Bind
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.btn_url.Bind(wx.EVT_BUTTON, self.on_open_url)
        self.btn_browse_root.Bind(wx.EVT_BUTTON, self.on_browse_root)
        self.btn_detect_ip.Bind(wx.EVT_BUTTON, self.on_detect_ip)
        self.chk_bind_addr.Bind(wx.EVT_CHECKBOX, self.update_bind_addr_state)
        self.txt_bind_addr.Bind(wx.EVT_TEXT, self.update_bind_addr_warning)
        self.chk_auth.Bind(wx.EVT_CHECKBOX, self.update_auth_state)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Update state
        self.load_settings()
        self.update_bind_addr_state()
        self.update_bind_addr_warning()
        self.update_auth_state()

    def on_close(self, event):
        self.save_settings()
        event.Skip()

        self.stop_event.set()
        if self.processing:
            max_wait = int(3 / 0.1)
            for _ in range(max_wait):
                if not self.stop_event.is_set():
                    break
                time.sleep(0.1)
            if self.stop_event.is_set():
                # It may be deadlocked, so force exit
                os._exit(-1)

    def on_browse_root(self, event):
        default_dir = self.txt_root.GetValue() or os.getcwd()
        with wx.DirDialog(
            self, T("Media Directory"), defaultPath=default_dir, style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_root.SetValue(dlg.GetPath())

    def on_detect_ip(self, event):
        self.txt_bind_addr.SetValue(get_local_address())

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

    def update_auth_state(self, *args, **kwargs):
        if self.chk_auth.IsChecked():
            self.txt_user.Enable()
            self.txt_password.Enable()
        else:
            self.txt_user.Disable()
            self.txt_password.Disable()

    def on_open_url(self, event):
        url = self.txt_url.GetValue()
        if url:
            start_file(url)

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
            None,
            message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
            caption=T("Error"),
            style=wx.OK,
        ) as dlg:
            dlg.ShowModal()

    def on_start(self, event):
        # Validate
        root = self.txt_root.GetValue()
        if not root or not path.isdir(root):
            wx.MessageBox(T("Please select a valid Media Directory."), T("Error"), wx.OK | wx.ICON_ERROR)
            return

        if not validate_number(self.txt_port.GetValue(), 1025, 65535, allow_empty=False):
            self.show_validation_error_message(T("Port"), 1025, 65535)
            return

        if self.chk_auth.IsChecked():
            user = self.txt_user.GetValue()
            password = self.txt_password.GetValue()
        else:
            user = password = None

        if self.chk_bind_addr.IsChecked():
            bind_addr = self.txt_bind_addr.GetAddress()
        else:
            bind_addr = get_local_address()

        parser = create_parser()
        parser.set_defaults(
            root=root, port=int(self.txt_port.GetValue()), bind_addr=bind_addr, user=user, password=password
        )
        args = parser.parse_args()

        self.stop_event.clear()
        set_state_args(args, stop_event=self.stop_event)

        # Update UI state
        self.btn_start.Disable()
        self.btn_stop.Enable()
        self.txt_root.Disable()
        self.btn_browse_root.Disable()

        url = f"https://{args.bind_addr}:{args.port}"
        self.txt_url.SetValue(url)
        self.btn_url.Enable()

        # Start worker thread
        self.processing = True
        startWorker(self.on_exit_worker, backend_main, wargs=(args,))

    def on_stop(self, event):
        self.stop_event.set()
        self.btn_stop.Disable()

    def on_exit_worker(self, result):
        self.processing = False
        self.btn_start.Enable()
        self.btn_stop.Disable()
        self.txt_root.Enable()
        self.btn_browse_root.Enable()
        self.txt_url.SetValue("")
        self.btn_url.Disable()

        try:
            result.get()
        except Exception:
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, T("Error"), wx.OK | wx.ICON_ERROR)

    def load_settings(self):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(manager, self)
        persistent_manager_restore_all(manager, exclude_names=["txt_url"])
        persistent_manager_unregister_all(manager)

    def save_settings(self):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(manager, self)
        manager.SaveAndUnregister()


LOCAL_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse
    import sys

    global LOCALE_DICT

    try:
        from .download_assets import main as download_main

        download_main()
    except ImportError:
        pass

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
    app = IW3PlayerApp()
    app.MainLoop()


if __name__ == "__main__":
    main()
