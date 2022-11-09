import inspect
import sys
from . model import Model
from .. logger import logger


_models = {}


def register_model(name, klass):
    global _models
    _models[name] = klass
    logger.debug("register %s -> %s", name, repr(klass))


def create_model(name, **kwargs):
    global _models
    if name not in _models:
        raise ValueError(f"Unknown model name: {name}")
    return _models[name](**kwargs)


def register_models(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            register_model(obj.name, obj)
