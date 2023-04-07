from .cunet import CUNet, UpCUNet
from .swin_unet import SwinUNet, SwinUNet2x, SwinUNet4x, SwinUNetDownscaled
from .vgg_7 import VGG7
from .upconv_7 import UpConv7
from .json_model import load_state_from_waifu2x_json
from .discriminator import (
    L3Discriminator, L3ConditionalDiscriminator,
    L3V1Discriminator, L3V1ConditionalDiscriminator,
)
import os
import importlib


__all__ = [
    "VGG7", "UpConv7",
    "CUNet", "UpCUNet",
    "SwinUNet", "SwinUNet2x", "SwinUNet4x", "SwinUNetDownscaled",
    "load_state_from_waifu2x_json"
]


# autoload for experimental models
# _*.py

_globals = globals()
for mod_file in os.listdir(os.path.dirname(__file__)):
    if (mod_file not in {"__init__.py", "__main__.py"} and
            mod_file.startswith("_") and mod_file.endswith(".py")):
        mod_name = mod_file[:-3]
        _globals[mod_name] = importlib.import_module('.' + mod_name, package=__name__)
