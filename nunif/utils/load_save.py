import json
import torch
from datetime import datetime, timezone
import torch.nn as nn
from .. models import create_model, Model


def load_state_from_waifu2x_json(model, json_file):
    with open(json_file, "r") as f:
        params = json.load(f)
    param_index = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)):
            param = params[param_index]
            param_index += 1
            if "weight" in param:
                m.weight.data.copy_(torch.FloatTensor(param["weight"]))
            if ("bias" in param) and m.bias is not None:
                m.bias.data.copy_(torch.FloatTensor(param["bias"]))
    return model


def save_model(model, model_path):
    if isinstance(model, nn.DataParallel):
        model = model.model
    assert(isinstance(model, Model))
    torch.save({"nunif_model": 1,
                "name": model.name,
                "updated_at": str(datetime.now(timezone.utc)),
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
