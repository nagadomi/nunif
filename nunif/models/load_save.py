import torch
import json
from datetime import datetime, timezone
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger


def load_state_from_waifu2x_json(model, json_file):
    logger.debug(f"load_state_from_waifu2x_json: {json_file}")
    with open(json_file, "r") as f:
        params = json.load(f)
    param_index = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)):
            param = params[param_index]
            param_index += 1
            if "weight" in param:
                w = torch.FloatTensor(param["weight"])
                logger.debug(f"w: {param['class_name']}{m.weight.shape} <- {w.shape}")
                m.weight.data.copy_(w.view(m.weight.shape))
            if ("bias" in param) and m.bias is not None:
                w = torch.FloatTensor(param["bias"])
                logger.debug(f"b: {param['class_name']}{m.weight.shape} <- {w.shape}")
                m.bias.data.copy_(w.view(m.bias.shape))
    logger.debug(f"read parameters: {len(params)}, write parameters: {param_index}")
    return model


def save_model(model, model_path, updated_at=None):
    if isinstance(model, nn.DataParallel):
        model = model.model
    assert(isinstance(model, Model))
    updated_at = str(updated_at or datetime.now(timezone.utc))
    torch.save({"nunif_model": 1,
                "name": model.name,
                "updated_at": updated_at,
                "kwargs": model._kwargs,
                "state_dict": model.state_dict()}, model_path)


def load_model(model_path):
    info = torch.load(model_path)
    assert("nunif_model" in info)
    model = create_model(info["name"], **info["kwargs"])
    model.load_state_dict(info["state_dict"])
    if "updated_at" in info:
        model.updated_at = info["updated_at"]
    return model
