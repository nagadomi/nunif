import torch
import sys
import os


_DISABLE_COMPILE = sys.platform != "linux" or os.getenv("NUNIF_DISABLE_COMPILE", False)


def compile(*args, **kwargs):
    if _DISABLE_COMPILE:
        return args[0]
    else:
        return torch.compile(*args, **kwargs)
