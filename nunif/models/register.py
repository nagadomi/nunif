import inspect
from torch import nn
from . model import Model
from .. logger import logger


_models = {}


def register_model(name, klass):
    global _models
    _models[name] = klass
    logger.debug("register %s -> %s", name, repr(klass))


def create_model(name, device_ids=None, **kwargs):
    logger.debug(f"create_model: {name}({kwargs})")
    global _models
    if name not in _models:
        raise ValueError(f"Unknown model name: {name}")
    model = _models[name](**kwargs)

    if device_ids is not None:
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            if device_ids[0] < 0:
                device = 'cpu'
            else:
                device = 'cuda:{}'.format(device_ids[0])
            model = model.to(device)

    return model


def get_model_names():
    return list(_models.keys())


def register_models(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            register_model(obj.name, obj)
