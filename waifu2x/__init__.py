import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

if "waifu2x.web" in getattr(sys, "orig_argv", []):
    os.environ["WAIFU2X_WEB"] = "1"

try:
    import truststore
    if not os.environ.get("NUNIF_TRUSTSTORE_INJECTED", False):
        truststore.inject_into_ssl()
        os.environ["NUNIF_TRUSTSTORE_INJECTED"] = "1"
except ModuleNotFoundError:
    pass

from .utils import Waifu2x
from . import models


__all__ = ["Waifu2x", "models"]
