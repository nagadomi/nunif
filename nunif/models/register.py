import inspect
from torch import nn
from . model import Model
from .. logger import logger
from .. device import create_device
from . data_parallel import DataParallelWrapper
import types


_models = {}


def _set(name, factory):
    global _models
    if name in _models:
        logger.warning("%s is already registered. Override.", name)
    _models[name] = factory
    if isinstance(factory, types.FunctionType):
        ident = factory.__name__
    else:
        ident = repr(factory)
    logger.debug("register %s -> %s", name, ident)


def _get(name):
    if name not in _models:
        raise ValueError(f"Unknown model name: {name}")
    return _models[name]


def register_model(cls):
    assert issubclass(cls, Model)
    _set(cls.name, cls)
    if hasattr(cls, "name_alias"):
        for alias in cls.name_alias:
            _set(alias, cls)
    return cls


def register_model_factory(name, func):
    _set(name, func)


def data_parallel_model(model, device_ids):
    if len(device_ids) > 1 and not isinstance(model, nn.DataParallel):
        model = DataParallelWrapper(model, device_ids=device_ids)
        return model
    else:
        return model


def create_model(name, device_ids=None, **kwargs):
    logger.debug(f"create_model: {name}({kwargs}), device_ids={device_ids}")
    model = _get(name)(**kwargs)

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


def _test():
    @register_model
    class AModel(Model):
        name = "A"

        def __init__(self, **kwargs):
            super(AModel, self).__init__({})

    def factory(**kwargs):
        return AModel(**kwargs)

    register_model_factory("B", factory)
    a = create_model("A")
    b = create_model("B")
    print(a)
    print(b)


if __name__ == "__main__":
    _test()
