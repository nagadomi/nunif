from nunif.initializer import init_torch_env

init_torch_env()


from . import models
from .utils import Waifu2x

__all__ = ["Waifu2x", "models"]
