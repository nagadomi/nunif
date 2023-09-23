import inspect
from torch import nn
from . model import Model
from .. logger import logger
from .. device import create_device


_models = {}


def register_model(cls):
    assert issubclass(cls, Model)
    global _models
    _models[cls.name] = cls
    logger.debug("register %s -> %s", cls.name, repr(cls))
    if hasattr(cls, "name_alias"):
        for alias in cls.name_alias:
            _models[alias] = cls
    return cls


def register_model_builder(name, func):
    global _models
    _models[name] = func
    logger.debug("register %s -> %s()", name, func.__name__)


def data_parallel_model(model, device_ids):
    if len(device_ids) > 1 and not isinstance(model, nn.DataParallel):
        name = model.name
        model = nn.DataParallel(model, device_ids=device_ids)
        # Set model name
        # TODO: this is a bad practice.
        setattr(model, "name", name)
        return model
    else:
        return model


def create_model(name, device_ids=None, **kwargs):
    logger.debug(f"create_model: {name}({kwargs}), device_ids={device_ids}")
    global _models
    if name not in _models:
        raise ValueError(f"Unknown model name: {name}")
    model = _models[name](**kwargs)

    if device_ids is not None:
        if len(device_ids) > 1:
            model = data_parallel_model(model, device_ids)
        else:
            device = create_device(device_ids)
            model = model.to(device)

    return model


def get_model_names():
    return list(_models.keys())


def register_models(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            register_model(obj)
