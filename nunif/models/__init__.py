import sys
from . model import Model
from . import waifu2x
from . load_save import load_model, save_model, load_state_from_waifu2x_json
from . register import register_model, create_model, register_models


register_models(sys.modules[__name__])
register_models(waifu2x)

__all__ = ["Model", "waifu2x",
           "load_model", "save_model", "load_state_from_waifu2x_json",
           "register_model", "create_model"]
