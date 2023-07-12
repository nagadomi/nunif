import sys
import os


if sys.executable.endswith("pythonw.exe"):
    """
    python code started from pythonw.exe crashes when accessing stdout/stderr.
    so reopen stdout/stderr with devnull.
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
