# Adapted from nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
# MIT License: Copyright (c) 2022 Andrej Karpathy
#
# Adding modules used in this repo by nagadomi
import torch
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2, ShiftedWindowAttention
from ..modules.norm import LayerNormNoBias, LayerNormNoBias2d


def configure_adamw(model, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
    )
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.Embedding,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm1d,
        torch.nn.GroupNorm,
        LayerNormNoBias,
        LayerNormNoBias2d,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif isinstance(m, torch.nn.MultiheadAttention):
                param_name = pn.split(".")[-1]
                if param_name in {"q_proj_weight", "k_proj_weight", "v_proj_weight", "in_proj_weight"}:
                    decay.add(fpn)
                else:
                    pass
            elif isinstance(m, ShiftedWindowAttentionV2):
                if pn.endswith("logit_scale"):
                    no_decay.add(fpn)
            elif isinstance(m, ShiftedWindowAttention):
                if pn.endswith("relative_position_bias_table"):
                    no_decay.add(fpn)
            elif m.__class__.__name__ in {"ParametrizedConv2d", "ParametrizedLinear",
                                          "ParametrizedConvTranspose2d"}:
                no_decay.add(fpn)
            else:
                pass

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer
