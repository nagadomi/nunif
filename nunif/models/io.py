import torch
from datetime import datetime, timezone
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger


def save_model(model, model_path, updated_at=None, train_kwargs=None):
    if isinstance(model, nn.DataParallel):
        model = model.model
    assert(isinstance(model, Model))
    updated_at = str(updated_at or datetime.now(timezone.utc))
    if train_kwargs is not None and not isinstance(train_kwargs, dict):
        train_kwargs = vars(train_kwargs) # Namespace
    torch.save({"nunif_model": 1,
                "name": model.name,
                "updated_at": updated_at,
                "kwargs": model.kwargs,
                "train_kwargs": train_kwargs,
                "state_dict": model.state_dict()}, model_path)


def load_model(model_path, strict=True, map_location="cpu"):
    data = torch.load(model_path, map_location=map_location)
    assert("nunif_model" in data)
    model = create_model(data["name"], **data["kwargs"])
    model.load_state_dict(data["state_dict"], strict=strict)
    logger.debug(f"load: {model.name} from {model_path}")
    if "updated_at" in data:
        model.updated_at = data["updated_at"]
    data.pop("state_dict")
    return model, data
