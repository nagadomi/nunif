from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import ParametrizationList
from torchvision.models.swin_transformer import ShiftedWindowAttention

from ..logger import logger
from ..modules.fusion import AdaptiveWeight, Lerp
from ..modules.norm import GroupNormNoBias, ReparamBatchNorm2d, RMSNorm1, RMSNorm2d
from ..optim import CompositeOptimizer, MuonWrapper

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

MUON_WEIGHT_MODULES: tuple[type[nn.Module], ...] = (nn.Linear,)
MUON_POINTWISE_CONV_MODULES: tuple[type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


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


def _iter_named_trainable_parameters_with_ancestors(
    model: nn.Module,
) -> Iterator[tuple[tuple[tuple[nn.Module, str], ...], nn.Module, str, str, nn.Parameter]]:
    """Yield (ancestors_inner_to_outer, owning_module, local_param_name, full_param_name, param).

    Each entry in ``ancestors_inner_to_outer`` is ``(ancestor_module, relative_param_name)``.
    """

    def _walk(
        module: nn.Module,
        prefix_parts: list[str],
        chain: list[nn.Module],
        visiting: set[int],
    ) -> Iterator[tuple[tuple[tuple[nn.Module, str], ...], nn.Module, str, str, nn.Parameter]]:
        mod_id = id(module)
        if mod_id in visiting:
            return
        visiting.add(mod_id)
        chain.append(module)
        try:
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_parts = [*prefix_parts, param_name]
                full_name = ".".join(full_parts)
                # Build (ancestor, relative_name) from innermost (owning module) to outermost (root).
                depth = len(chain)
                ancestors = tuple((chain[i], ".".join(full_parts[i:])) for i in range(depth - 1, -1, -1))
                yield ancestors, module, param_name, full_name, param

            for child_name, child in module.named_children():
                prefix_parts.append(child_name)
                try:
                    yield from _walk(child, prefix_parts, chain, visiting)
                finally:
                    prefix_parts.pop()
        finally:
            chain.pop()
            visiting.remove(mod_id)

    yield from _walk(model, [], [], set())


def _resolve_config_hook(
    ancestors: tuple[tuple[nn.Module, str], ...],
    hook_name: str,
) -> bool | None:
    for module, rel_name in ancestors:
        hook = getattr(module, hook_name, None)
        if callable(hook):
            ret = hook(rel_name)
            if ret is not None:
                if not isinstance(ret, bool):
                    raise TypeError(
                        f"{module.__class__.__name__}.{hook_name}({rel_name!r}) "
                        f"must return bool or None, got {type(ret).__name__}"
                    )
                return ret
    return None


def _is_muon_compatible_shape(param: nn.Parameter) -> bool:
    return param.ndim == 2 or (param.ndim >= 3 and all(s == 1 for s in param.shape[2:]))


def should_apply_weight_decay(
    module: nn.Module,
    param_name: str,
    *,
    ancestors: tuple[tuple[nn.Module, str], ...] | None = None,
) -> bool | None:
    """Return True for decay, False for no decay, or None if unclassified.

    Any owning or ancestor module may define ``weight_decay_config(param_name: str) -> bool | None``
    using the parameter name relative to that module (innermost non-``None`` result wins).
    Returning ``None`` falls back to outer ancestors and then the default rules.
    Parametrized weights owned by ``ParametrizationList`` (e.g. ``spectral_norm``)
    are excluded from weight decay by default.
    """
    chain = ancestors if ancestors is not None else ((module, param_name),)
    hook_decision = _resolve_config_hook(chain, "weight_decay_config")
    if hook_decision is not None:
        return hook_decision

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

    for ancestors, module, param_name, full_name, param in _iter_named_trainable_parameters_with_ancestors(model):
        decision = should_apply_weight_decay(module, param_name, ancestors=ancestors)
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


def should_use_muon(
    module: nn.Module,
    param_name: str,
    param: nn.Parameter,
    *,
    ancestors: tuple[tuple[nn.Module, str], ...] | None = None,
) -> bool:
    """Return True if a decay-eligible 2D or pointwise conv weight should be optimized by Muon.

    Any owning or ancestor module may define ``muon_config(param_name: str) -> bool | None``
    using the parameter name relative to that module (innermost non-``None`` result wins).
    """
    chain = ancestors if ancestors is not None else ((module, param_name),)
    hook_decision = _resolve_config_hook(chain, "muon_config")
    if hook_decision is not None:
        if hook_decision and not _is_muon_compatible_shape(param):
            raise ValueError(
                f"muon_config enabled Muon for incompatible parameter shape {tuple(param.shape)} "
                f"on {module.__class__.__name__}.{param_name}"
            )
        return hook_decision

    if isinstance(module, ParametrizationList):
        return False
    if param.ndim == 2:
        if param_name == "weight" and isinstance(module, MUON_WEIGHT_MODULES):
            return True
        if isinstance(module, nn.MultiheadAttention) and param_name in _MHA_DECAY_PARAMS:
            return True
        if isinstance(module, (nn.RNNBase, nn.RNNCellBase)) and param_name.startswith("weight_"):
            return True
        return False
    if (
        param_name == "weight"
        and isinstance(module, MUON_POINTWISE_CONV_MODULES)
        and module.groups == 1
        and all(k == 1 for k in module.kernel_size)
        and _is_muon_compatible_shape(param)
    ):
        return True
    return False


def split_muon_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into (muon_decay, adamw_decay, adamw_no_decay) sorted by full name."""
    decisions: dict[int, tuple[bool, bool, str, nn.Parameter]] = {}
    unclassified: list[str] = []
    conflicts: list[str] = []

    for ancestors, module, param_name, full_name, param in _iter_named_trainable_parameters_with_ancestors(model):
        decay = should_apply_weight_decay(module, param_name, ancestors=ancestors)
        if decay is None:
            unclassified.append(f"{full_name} ({module.__class__.__name__}, shape={tuple(param.shape)})")
            continue

        muon_hook = _resolve_config_hook(ancestors, "muon_config")
        if muon_hook is not None:
            if muon_hook and not decay:
                raise ValueError(f"muon_config enabled Muon for non-decay parameter: {full_name}")
            if muon_hook and not _is_muon_compatible_shape(param):
                raise ValueError(
                    f"muon_config enabled Muon for incompatible parameter shape {tuple(param.shape)} on {full_name}"
                )
            use_muon = decay and muon_hook
        else:
            use_muon = decay and should_use_muon(module, param_name, param, ancestors=())
        param_id = id(param)
        prev = decisions.get(param_id)
        if prev is None:
            decisions[param_id] = (decay, use_muon, full_name, param)
        elif prev[0] != decay or prev[1] != use_muon:
            conflicts.append(
                f"{prev[2]} (decay={prev[0]}, muon={prev[1]}) vs {full_name} (decay={decay}, muon={use_muon})"
            )
        elif full_name < prev[2]:
            decisions[param_id] = (decay, use_muon, full_name, param)

    if unclassified:
        raise ValueError("Unclassified parameters for weight decay: " + ", ".join(unclassified))
    if conflicts:
        raise ValueError("Conflicting optimizer decisions for tied parameters: " + ", ".join(conflicts))

    muon_decay_named = sorted(
        ((name, param) for decay, use_muon, name, param in decisions.values() if decay and use_muon),
        key=lambda item: item[0],
    )
    adamw_decay_named = sorted(
        ((name, param) for decay, use_muon, name, param in decisions.values() if decay and not use_muon),
        key=lambda item: item[0],
    )
    adamw_no_decay_named = sorted(
        ((name, param) for decay, _, name, param in decisions.values() if not decay),
        key=lambda item: item[0],
    )
    return (
        [param for _, param in muon_decay_named],
        [param for _, param in adamw_decay_named],
        [param for _, param in adamw_no_decay_named],
    )


def configure_muon(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    nesterov: bool = True,
    adjust_lr_fn: str | None = "match_rms_adamw",
    betas: tuple[float, float] = (0.9, 0.999),
    adamw_lr: float | None = None,
    adamw_weight_decay: float | None = None,
    fused: bool | None = None,
) -> CompositeOptimizer:
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch version")

    muon_decay_params, adamw_decay_params, adamw_no_decay_params = split_muon_params(model)
    aux_lr = adamw_lr if adamw_lr is not None else lr
    aux_wd = adamw_weight_decay if adamw_weight_decay is not None else weight_decay

    optimizers: list[torch.optim.Optimizer] = []
    if muon_decay_params:
        optimizers.append(
            MuonWrapper(
                [{"params": muon_decay_params, "weight_decay": weight_decay}],
                lr=lr,
                weight_decay=0.0,
                momentum=momentum,
                nesterov=nesterov,
                adjust_lr_fn=adjust_lr_fn,
            )
        )

    adamw_groups: list[dict[str, Any]] = []
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": aux_wd})
    if adamw_no_decay_params:
        adamw_groups.append({"params": adamw_no_decay_params, "weight_decay": 0.0})

    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(
                adamw_groups,
                lr=aux_lr,
                betas=betas,
                weight_decay=0.0,
                fused=fused,
            )
        )

    if not optimizers:
        raise ValueError("optimizer got an empty parameter list")

    if not muon_decay_params:
        logger.warning("No Muon-eligible 2D parameters found; falling back to AdamW only")

    return CompositeOptimizer(optimizers)
