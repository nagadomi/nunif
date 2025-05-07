from packaging import version as packaging_version
import torch
from datetime import datetime, timezone
from collections import OrderedDict
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger
from .. device import create_device, autocast


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
        map_location = "cpu"
    if model_path.startswith("http://") or model_path.startswith("https://"):
        # force weights_only=True to avoid security risk
        data = torch.hub.load_state_dict_from_url(model_path, weights_only=True, map_location=map_location)
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
    if hasattr(model, "get_device"):
        return model.get_device()
    else:
        return next(model.parameters()).device


_COMPILER_SUPPORTED_DEVICES = {}


def check_compile_support(device):
    device_name = device if isinstance(device, str) else device.type
    if device_name not in _COMPILER_SUPPORTED_DEVICES:
        try:
            model = torch.nn.Linear(32, 32, bias=False)
            model.weight.data.zero_()
            model = torch.compile(model.eval().to(device))
            with torch.inference_mode(), autocast(device):
                model(torch.zeros((1, 32), device=device))
            _COMPILER_SUPPORTED_DEVICES[device_name] = True
        except:  # noqa  #(RuntimeError, AssertionError):
            # import sys
            # print(device_name, sys.exc_info())
            _COMPILER_SUPPORTED_DEVICES[device_name] = False

    return _COMPILER_SUPPORTED_DEVICES[device_name]


def compile_model(model, **kwargs):
    if not is_compiled_model(model) and check_compile_support(get_model_device(model)):
        logger.debug(f"compile {model.__class__.__name__}, kwargs={kwargs}")
        model = torch.compile(model, **kwargs)
        setattr(model, "__nunif_compiled_model", True)
    return model


def is_compiled_model(model):
    # TODO: class name of compiled model is unclear
    return hasattr(model, "__nunif_compiled_model")


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


def _test_compile():
    model = torch.nn.Linear(32, 32, bias=False).cuda()
    print(is_compiled_model(model))
    model = compile_model(model)
    print(is_compiled_model(model))
    model(torch.zeros(1, 32).cuda())
    print(is_compiled_model(model))

    print(check_compile_support("cpu"))
    print(check_compile_support(torch.device("cpu")))
    print(check_compile_support(torch.device("cuda")))
    print(check_compile_support(torch.device("cuda:1")))
    print(check_compile_support("cuda:0"))
    print(check_compile_support("cuda:10"))
    print(check_compile_support("mps"))
    print(check_compile_support("xpu"))
    print(_COMPILER_SUPPORTED_DEVICES)


if __name__ == "__main__":
    _test_compile()
