import sys
import inspect
from . model import Model
from . import waifu2x
from .. utils.logger import logger

_models = {}


def register_model(name, klass):
    global _models
    _models[name] = klass


def create_model(name, **kwargs):
    global _models
    assert(name in _models)
    return _models[name](**kwargs)


def _register_models(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            logger.debug("register %s -> %s", obj.name, repr(obj))
            register_model(obj.name, obj)


_register_models(sys.modules[__name__])
_register_models(waifu2x)
