import json
import torch
import torch.nn as nn
from nunif.logger import logger


def load_state_from_waifu2x_json(model, json_file, skip_upsample_weight=False):
    logger.debug(f"load_state_from_waifu2x_json: {json_file}")
    with open(json_file, "r") as f:
        params = json.load(f)
    param_index = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)):
            param = params[param_index]
            param_index += 1
            if (skip_upsample_weight and
                ("SpatialFullConvolution" in param["class_name"]) and
                    ("weight" in param)):
                w = torch.FloatTensor(param["weight"])
                if len(w.shape) == 4 and w.shape[2] == 2 and w.shape[2] == 2:
                    logger.debug(f"skip : {param['class_name']}{w.shape}")
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
