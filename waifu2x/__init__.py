import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import truststore
truststore.inject_into_ssl()

from .utils import Waifu2x
from . import models


__all__ = ["Waifu2x", "models"]
