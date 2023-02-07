import os
import importlib
from .torchvision_models import VGG11BN

__all__ = ["VGG11BN"]


# autoload for experimental models
# _*.py

_globals = globals()
for mod_file in os.listdir(os.path.dirname(__file__)):
    if (mod_file not in {"__init__.py", "__main__.py"} and
            mod_file.startswith("_") and mod_file.endswith(".py")):
        mod_name = mod_file[:-3]
        _globals[mod_name] = importlib.import_module('.' + mod_name, package=__name__)
