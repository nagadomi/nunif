import torch
import sys


_DISABLE_COMPILE = sys.platform != "linux"


def compile(*args):
    if _DISABLE_COMPILE:
        return args[0]
    else:
        return torch.compile(*args)
