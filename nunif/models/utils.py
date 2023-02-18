import torch
from datetime import datetime, timezone
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger


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


def load_model(model_path, model=None, device_ids=None, strict=True, map_location="cpu"):
    data = torch.load(model_path, map_location=map_location)
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
        if device_ids[0] < 0:
            device = 'cpu'
        else:
            device = 'cuda:{}'.format(device_ids[0])
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


def to_data_parallel(model, device_ids):
    name = model.name
    model = nn.DataParallel(model, device_ids=device_ids)
    # Set model name
    # TODO: this is a bad practice.
    setattr(model, "name", name)
    return model
