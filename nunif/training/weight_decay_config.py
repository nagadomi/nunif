from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import ParametrizationList
from torchvision.models.swin_transformer import ShiftedWindowAttention

from ..modules.fusion import AdaptiveWeight, Lerp
from ..modules.norm import GroupNormNoBias, ReparamBatchNorm2d, RMSNorm1, RMSNorm2d

# RMSNorm1 is a subclass of nn.RMSNorm (0-centered), so DECAY_WEIGHT_MODULES
# must be checked before NO_DECAY_WEIGHT_MODULES.
DECAY_WEIGHT_MODULES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Bilinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    RMSNorm1,
    RMSNorm2d,
)

NO_DECAY_WEIGHT_MODULES: tuple[type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.RMSNorm,
    nn.Embedding,
    nn.EmbeddingBag,
    nn.PReLU,
    GroupNormNoBias,
    ReparamBatchNorm2d,
    Lerp,
    AdaptiveWeight,
)

_MHA_DECAY_PARAMS = frozenset({"q_proj_weight", "k_proj_weight", "v_proj_weight", "in_proj_weight"})
_MHA_NO_DECAY_PARAMS = frozenset({"bias_k", "bias_v"})
_SWA_NO_DECAY_PARAMS = frozenset({"logit_scale", "relative_position_bias_table"})


def iter_named_trainable_parameters(
    model: nn.Module,
    *,
    deduplicate: bool = True,
) -> Iterator[tuple[nn.Module, str, str, nn.Parameter]]:
    """Yield (owning_module, local_param_name, full_param_name, param) for trainable parameters."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if deduplicate:
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            yield module, param_name, full_name, param


def should_apply_weight_decay(module: nn.Module, param_name: str) -> bool | None:
    """Return True for decay, False for no decay, or None if unclassified.

    A module may define ``weight_decay_config(param_name: str) -> bool | None`` to
    override classification for its own direct parameters (local names only; it does
    not apply to child modules). Returning ``None`` falls back to the default rules.
    Parametrized weights owned by ``ParametrizationList`` (e.g. ``spectral_norm``)
    are unconditionally excluded from weight decay.
    """
    weight_decay_config = getattr(module, "weight_decay_config", None)
    if callable(weight_decay_config):
        ret = weight_decay_config(param_name)
        if ret is not None:
            if not isinstance(ret, bool):
                raise TypeError(
                    f"{module.__class__.__name__}.weight_decay_config({param_name!r}) "
                    f"must return bool or None, got {type(ret).__name__}"
                )
            return ret

    if isinstance(module, ParametrizationList):
        return False

    if param_name.endswith("bias"):
        return False

    if isinstance(module, nn.MultiheadAttention):
        if param_name in _MHA_DECAY_PARAMS:
            return True
        if param_name in _MHA_NO_DECAY_PARAMS:
            return False

    if isinstance(module, ShiftedWindowAttention) and param_name in _SWA_NO_DECAY_PARAMS:
        return False

    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
        if param_name.startswith("weight_"):
            return True
        if param_name.startswith("bias_"):
            return False

    if param_name == "weight":
        if isinstance(module, DECAY_WEIGHT_MODULES):
            return True
        if isinstance(module, NO_DECAY_WEIGHT_MODULES):
            return False

    return None


def split_weight_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters of a model into (decay_params, no_decay_params) sorted by full name."""
    decisions: dict[int, tuple[bool, str, nn.Parameter]] = {}
    unclassified: list[str] = []
    conflicts: list[str] = []

    for module, param_name, full_name, param in iter_named_trainable_parameters(model, deduplicate=False):
        decision = should_apply_weight_decay(module, param_name)
        if decision is None:
            unclassified.append(f"{full_name} ({module.__class__.__name__}, shape={tuple(param.shape)})")
            continue

        param_id = id(param)
        prev = decisions.get(param_id)
        if prev is None:
            decisions[param_id] = (decision, full_name, param)
        elif prev[0] != decision:
            conflicts.append(f"{prev[1]} (decay={prev[0]}) vs {full_name} (decay={decision})")
        elif full_name < prev[1]:
            decisions[param_id] = (decision, full_name, param)

    if unclassified:
        raise ValueError("Unclassified parameters for weight decay: " + ", ".join(unclassified))
    if conflicts:
        raise ValueError("Conflicting weight decay decisions for tied parameters: " + ", ".join(conflicts))

    decay_named = sorted(
        ((name, param) for decision, name, param in decisions.values() if decision),
        key=lambda item: item[0],
    )
    no_decay_named = sorted(
        ((name, param) for decision, name, param in decisions.values() if not decision),
        key=lambda item: item[0],
    )
    return [param for _, param in decay_named], [param for _, param in no_decay_named]


def configure_optim_groups(model: nn.Module, weight_decay: float = 0.01) -> list[dict[str, Any]]:
    decay_params, no_decay_params = split_weight_decay_params(model)
    optim_groups: list[dict[str, Any]] = []
    if decay_params:
        optim_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        optim_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return optim_groups


def configure_adamw(
    model: nn.Module,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    optim_groups = configure_optim_groups(model, weight_decay=weight_decay)
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, weight_decay=0.0, fused=fused)
