from nunif.initializer import init_torch_env

init_torch_env()

from . import models  # noqa: F401, E402
from .class_names import CLASS_NAMES  # noqa: E402

__all__ = ["CLASS_NAMES"]
