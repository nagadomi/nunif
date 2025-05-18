# This code patches the subprocess.Popen constructor on Windows
# so that every time a new process is started via subprocess,
# it automatically includes CREATE_NO_WINDOW flag.
#
# Background:
# On Windows, when running a Python GUI application (e.g., with pythonw.exe)
# that uses torch.compile(),
# the compiler (cl.exe) may be invoked in the background.
# Each of these invocations can cause a black console window to flash on the screen.
import subprocess
import sys


if sys.platform == "win32":
    CREATE_NO_WINDOW = 0x08000000
    _original_popen_init = subprocess.Popen.__init__

    def patched_popen_init(self, *args, **kwargs):
        creationflags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = creationflags | CREATE_NO_WINDOW
        return _original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_popen_init
