import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if "waifu2x.web" in getattr(sys, "orig_argv", []):
    os.environ["WAIFU2X_WEB"] = "1"


from . import models
from .utils import Waifu2x

__all__ = ["Waifu2x", "models"]
