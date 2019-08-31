import inspect
from . model import Model
from .. logger import logger


_models = {}


def register_model(name, klass):
    global _models
    _models[name] = klass


def create_model(name, **kwargs):
    global _models
    assert(name in _models)
    return _models[name](**kwargs)


def register_models(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            logger.debug("register %s -> %s", obj.name, repr(obj))
            register_model(obj.name, obj)
