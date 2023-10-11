from packaging import version as packaging_version
import sys
import torch
from datetime import datetime, timezone
from collections import OrderedDict
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger
from .. device import create_device


PYTORCH2 = packaging_version.parse(torch.__version__).major >= 2


def save_model(model, model_path, updated_at=None, train_kwargs=None, **kwargs):
    if isinstance(model, nn.DataParallel):
        model = model.module
    assert (isinstance(model, Model))
    updated_at = str(updated_at or datetime.now(timezone.utc))
    if train_kwargs is not None:
        if not isinstance(train_kwargs, dict):
            # Namespace to dict
            train_kwargs = vars(train_kwargs)
        # Remove data that probably cannot be loaded in other environments
        # The current intended target is a handler
        remove_keys = [k for k in train_kwargs.keys()
                       if callable(train_kwargs[k])]
        for k in remove_keys:
            train_kwargs.pop(k)

    data = {
        "nunif_model": 1,
        "name": model.name,
        "updated_at": updated_at,
        "kwargs": model.get_kwargs(),
        "train_kwargs": train_kwargs,
        "state_dict": model.state_dict()}
    data.update(kwargs)
    torch.save(data, model_path)


def load_model(model_path, model=None, device_ids=None,
               strict=True, map_location="cpu", weights_only=False):
    if not PYTORCH2:
        # Disabled due to https://github.com/pytorch/pytorch/issues/94670
        weights_only = False
    if "mps" in str(map_location):
        data = torch.load(model_path, map_location="cpu", weights_only=weights_only)
    else:
        data = torch.load(model_path, map_location=map_location, weights_only=weights_only)
    assert ("nunif_model" in data)
    if model is None:
        model = create_model(data["name"], device_ids=device_ids, **data["kwargs"])
        model_predefine = False
    else:
        model_predefine = True
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(data["state_dict"], strict=strict)
    else:
        model.load_state_dict(data["state_dict"], strict=strict)
    logger.debug(f"load: {model.name} from {model_path}")
    if "updated_at" in data:
        model.updated_at = data["updated_at"]
    data.pop("state_dict")

    if not model_predefine and device_ids is not None:
        device = create_device(device_ids)
        model = model.to(device)

    return model, data


def get_model_config(model, key=None):
    if isinstance(model, nn.DataParallel):
        model = model.module
    config = model.get_config()
    if key is None:
        return config
    else:
        return config[key]


def get_model_kwargs(model, key=None):
    if isinstance(model, nn.DataParallel):
        model = model.module
    kwargs = model.get_kwargs()
    if key is None:
        return kwargs
    else:
        return kwargs[key]


def get_model_device(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return model.get_device()


def call_model_method(model, name, **kwargs):
    if isinstance(model, nn.DataParallel):
        model = model.model
    func = getattr(model, name, None)
    if not (func is not None and callable(func)):
        raise ValueError(f"Unable to call {type(model)}.{name}")

    return func(**kwargs)


def compile_model(model, **kwargs):
    # Windows not yet supported for torch.compile
    if PYTORCH2 and sys.platform == "linux" and not is_compiled_model(model):
        # only cuda
        if get_model_device(model).type == "cuda":
            model = torch.compile(model, **kwargs)
    return model


def is_compiled_model(model):
    return not isinstance(model, Model)


def merge_state_dict(a, b, alpha=0.5):
    """
    NOTE: This only works when `a` and `b` are finetuned models of the same original model.
          Also constraints may be broken. Should always be verified to work.
    """
    assert a.keys() == b.keys()
    c = OrderedDict()
    for k in a.keys():
        c[k] = a[k] * alpha + b[k] * (1. - alpha)
    return c


def mean_state_dict(dicts):
    assert len(dicts) > 0
    a = dicts[0]
    assert all(a.keys() == d.keys() for d in dicts)
    mean = OrderedDict()
    scale = 1. / len(dicts)
    for k in a.keys():
        for d in dicts:
            if k not in mean:
                mean[k] = d[k] * scale
            else:
                mean[k] += d[k] * scale
    return mean


class DataParallelWrapper(nn.DataParallel):
    # ref: https://discuss.pytorch.org/t/making-a-wrapper-around-nn-dataparallel-to-access-module-attributes-is-safe/79124
    def __init__(self, module, device_ids=None):
        super().__init__(module, device_ids=device_ids)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
