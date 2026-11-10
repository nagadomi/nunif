import os
import sys

from nunif.initializer import init_torch_env

init_torch_env()
if "waifu2x.web" in getattr(sys, "orig_argv", []):
    os.environ["WAIFU2X_WEB"] = "1"


from . import models
from .utils import Waifu2x

__all__ = ["Waifu2x", "models"]
