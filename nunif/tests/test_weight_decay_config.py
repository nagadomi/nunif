import pytest
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models.swin_transformer import ShiftedWindowAttention, ShiftedWindowAttentionV2

from nunif.modules.fusion import AdaptiveWeight, AdaptiveWeightedAdd, Lerp
from nunif.modules.norm import GroupNormNoBias, LayerNormNoBias2d, ReparamBatchNorm2d, RMSNorm1, RMSNorm2d
from nunif.training.cosine_wd import CosineAnnealingWarmRestartsWithFixedWeightDecay
from nunif.training.weight_decay_config import (
    configure_adamw,
    configure_optim_groups,
    split_weight_decay_params,
)


def _param_ids(params: list[nn.Parameter]) -> set[int]:
    return {id(p) for p in params}


def test_standard_modules() -> None:
    model = nn.ModuleDict(
        {
            "linear": nn.Linear(4, 4),
            "bilinear": nn.Bilinear(4, 4, 4),
            "conv1d": nn.Conv1d(4, 4, 3),
            "conv2d": nn.Conv2d(4, 4, 3),
            "conv3d": nn.Conv3d(4, 4, 3),
            "convt1d": nn.ConvTranspose1d(4, 4, 2),
            "convt2d": nn.ConvTranspose2d(4, 4, 2),
            "convt3d": nn.ConvTranspose3d(4, 4, 2),
            "ln": nn.LayerNorm(4),
            "bn1d": nn.BatchNorm1d(4),
            "bn2d": nn.BatchNorm2d(4),
            "bn3d": nn.BatchNorm3d(4),
            "gn": nn.GroupNorm(2, 4),
            "in1d": nn.InstanceNorm1d(4, affine=True),
            "in2d": nn.InstanceNorm2d(4, affine=True),
            "in3d": nn.InstanceNorm3d(4, affine=True),
            "rmsnorm": nn.RMSNorm(4),
            "emb": nn.Embedding(8, 4),
            "emb_bag": nn.EmbeddingBag(8, 4),
            "prelu": nn.PReLU(4),
        }
    )

    decay, no_decay = split_weight_decay_params(model)
    decay_ids = _param_ids(decay)
    no_decay_ids = _param_ids(no_decay)

    for key in ("linear", "bilinear", "conv1d", "conv2d", "conv3d", "convt1d", "convt2d", "convt3d"):
        mod = model[key]
        assert id(mod.weight) in decay_ids
        assert id(mod.bias) in no_decay_ids

    for key in ("ln", "bn1d", "bn2d", "bn3d", "gn", "in1d", "in2d", "in3d", "rmsnorm", "emb", "emb_bag", "prelu"):
        mod = model[key]
        assert id(mod.weight) in no_decay_ids
        if getattr(mod, "bias", None) is not None:
            assert id(mod.bias) in no_decay_ids


def test_rnn_modules() -> None:
    lstm = nn.LSTM(4, 4, num_layers=2)
    gru_cell = nn.GRUCell(4, 4)
    model = nn.ModuleDict({"lstm": lstm, "gru_cell": gru_cell})

    decay, no_decay = split_weight_decay_params(model)
    decay_ids = _param_ids(decay)
    no_decay_ids = _param_ids(no_decay)

    for name, param in model.named_parameters():
        leaf_name = name.split(".")[-1]
        if leaf_name.startswith("weight_"):
            assert id(param) in decay_ids
        elif leaf_name.startswith("bias_"):
            assert id(param) in no_decay_ids
        else:
            pytest.fail(f"Unexpected RNN parameter: {name}")


def test_nunif_custom_modules() -> None:
    rms1 = RMSNorm1(4)
    rms2d = RMSNorm2d(4)
    ln_nobias2d = LayerNormNoBias2d(4)
    gn_nobias = GroupNormNoBias(2, 4)
    reparam_bn = ReparamBatchNorm2d(4)
    lerp = Lerp()
    adaptive_weight = AdaptiveWeight(3)
    adaptive_weighted_add = AdaptiveWeightedAdd(4, 3)

    model = nn.ModuleDict(
        {
            "rms1": rms1,
            "rms2d": rms2d,
            "ln_nobias2d": ln_nobias2d,
            "gn_nobias": gn_nobias,
            "reparam_bn": reparam_bn,
            "lerp": lerp,
            "adaptive_weight": adaptive_weight,
            "adaptive_weighted_add": adaptive_weighted_add,
        }
    )

    decay, no_decay = split_weight_decay_params(model)
    decay_ids = _param_ids(decay)
    no_decay_ids = _param_ids(no_decay)

    assert id(rms1.weight) in decay_ids
    assert id(rms2d.weight) in decay_ids

    assert id(ln_nobias2d.norm.weight) in no_decay_ids
    assert id(gn_nobias.weight) in no_decay_ids
    assert id(reparam_bn.weight) in no_decay_ids
    assert id(reparam_bn.bias) in no_decay_ids
    assert id(lerp.weight) in no_decay_ids
    assert id(adaptive_weight.weight) in no_decay_ids
    assert id(adaptive_weighted_add.adaptive_weight.weight) in no_decay_ids


def test_spectral_norm_modules() -> None:
    model = nn.Sequential(
        spectral_norm(nn.Conv1d(4, 4, 3)),
        spectral_norm(nn.Conv2d(4, 4, 3)),
        spectral_norm(nn.ConvTranspose2d(4, 4, 2)),
        spectral_norm(nn.Linear(4, 4)),
    )

    decay, no_decay = split_weight_decay_params(model)
    assert len(decay) == 0
    assert len(no_decay) == len(list(model.parameters()))


def test_attention_modules() -> None:
    mha_same = nn.MultiheadAttention(embed_dim=8, num_heads=2, add_bias_kv=True)
    mha_diff = nn.MultiheadAttention(embed_dim=8, num_heads=2, kdim=4, vdim=4)
    swa = ShiftedWindowAttention(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    swa_v2 = ShiftedWindowAttentionV2(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    model = nn.ModuleDict(
        {
            "mha_same": mha_same,
            "mha_diff": mha_diff,
            "swa": swa,
            "swa_v2": swa_v2,
        }
    )

    decay, no_decay = split_weight_decay_params(model)
    decay_ids = _param_ids(decay)
    no_decay_ids = _param_ids(no_decay)

    assert id(mha_same.in_proj_weight) in decay_ids
    assert id(mha_same.in_proj_bias) in no_decay_ids
    assert id(mha_same.bias_k) in no_decay_ids
    assert id(mha_same.bias_v) in no_decay_ids
    assert id(mha_diff.q_proj_weight) in decay_ids
    assert id(mha_diff.k_proj_weight) in decay_ids
    assert id(mha_diff.v_proj_weight) in decay_ids
    assert id(swa.relative_position_bias_table) in no_decay_ids
    assert id(swa_v2.logit_scale) in no_decay_ids


def test_custom_weight_decay_config_hook_does_not_leak_to_children() -> None:
    class ParentWithHook(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))
            self.extra = nn.Parameter(torch.ones(2, 2))
            self.linear = nn.Linear(4, 4)

        def weight_decay_config(self, pn: str) -> bool | None:
            if pn == "scale":
                return False
            if pn == "extra":
                return True
            return None

    model = ParentWithHook()
    decay, no_decay = split_weight_decay_params(model)
    decay_ids = _param_ids(decay)
    no_decay_ids = _param_ids(no_decay)

    assert id(model.scale) in no_decay_ids
    assert id(model.extra) in decay_ids
    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids


def test_custom_weight_decay_config_hook_overrides_defaults_and_validates_type() -> None:
    class CustomLinear(nn.Linear):
        def weight_decay_config(self, pn: str) -> bool | None:
            if pn == "weight":
                return False
            if pn == "bias":
                return True
            return None

    model = CustomLinear(4, 4)
    decay, no_decay = split_weight_decay_params(model)
    assert _param_ids(decay) == {id(model.bias)}
    assert _param_ids(no_decay) == {id(model.weight)}

    class InvalidHookModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def weight_decay_config(self, pn: str) -> str:
            return "no_decay"

    with pytest.raises(TypeError, match="must return bool or None"):
        split_weight_decay_params(InvalidHookModule())


def test_frozen_parameters_excluded() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[0].requires_grad_(False)

    decay, no_decay = split_weight_decay_params(model)
    assert _param_ids(decay) == {id(model[1].weight)}
    assert _param_ids(no_decay) == {id(model[1].bias)}

    model.requires_grad_(False)
    assert configure_optim_groups(model) == []
    with pytest.raises(ValueError):
        configure_adamw(model)


def test_tied_weights_deduplicated_and_conflicts_detected() -> None:
    class TiedLinearModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4, bias=False)
            self.fc2 = nn.Linear(4, 4, bias=False)
            self.fc2.weight = self.fc1.weight

    model = TiedLinearModel()
    decay, no_decay = split_weight_decay_params(model)
    assert len(decay) == 1
    assert decay[0] is model.fc1.weight
    assert len(no_decay) == 0

    class ConflictingTiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(4, 4)
            self.head = nn.Linear(4, 4, bias=False)
            self.head.weight = self.emb.weight

    with pytest.raises(ValueError, match="Conflicting weight decay decisions"):
        split_weight_decay_params(ConflictingTiedModel())


def test_parameter_order_is_sorted_by_full_name() -> None:
    class UnsortedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.z_fc = nn.Linear(4, 4)
            self.a_fc = nn.Linear(4, 4)

    model = UnsortedModel()
    decay, no_decay = split_weight_decay_params(model)
    assert decay == [model.a_fc.weight, model.z_fc.weight]
    assert no_decay == [model.a_fc.bias, model.z_fc.bias]


def test_data_parallel_wrapped_model() -> None:
    base = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    wrapped = nn.DataParallel(base)
    decay, no_decay = split_weight_decay_params(wrapped)
    assert _param_ids(decay) == {id(base[0].weight)}
    assert _param_ids(no_decay) == {id(base[0].bias), id(base[1].weight), id(base[1].bias)}


def test_configure_optim_groups_omits_empty_groups() -> None:
    decay_only = nn.Linear(4, 4, bias=False)
    groups = configure_optim_groups(decay_only, weight_decay=0.05)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.05

    no_decay_only = nn.LayerNorm(4)
    groups = configure_optim_groups(no_decay_only, weight_decay=0.05)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.0


def test_unclassified_parameter_raises_value_error() -> None:
    class UnknownParamModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery = nn.Parameter(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="mystery.*UnknownParamModule"):
        split_weight_decay_params(UnknownParamModule())


def test_configure_adamw_and_fixed_wd_scheduler() -> None:
    model = nn.Linear(4, 4)
    optimizer = configure_adamw(model, lr=0.01, weight_decay=0.001)
    assert len(optimizer.param_groups) == 2
    assert optimizer.defaults["weight_decay"] == 0.0

    scheduler = CosineAnnealingWarmRestartsWithFixedWeightDecay(optimizer, T_0=10, eta_min=0.001)
    assert optimizer.param_groups[0]["_initial_weight_decay"] == 0.001
    assert "_initial_weight_decay" not in optimizer.param_groups[1]

    scheduler.step()
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    # Verify resume via state_dict / load_state_dict preserves _initial_weight_decay.
    opt_state = optimizer.state_dict()
    sched_state = scheduler.state_dict()

    resumed_optimizer = configure_adamw(model, lr=0.01, weight_decay=0.001)
    resumed_scheduler = CosineAnnealingWarmRestartsWithFixedWeightDecay(resumed_optimizer, T_0=10, eta_min=0.001)
    resumed_optimizer.load_state_dict(opt_state)
    resumed_scheduler.load_state_dict(sched_state)
    assert resumed_optimizer.param_groups[0]["_initial_weight_decay"] == 0.001
    resumed_scheduler.step()
    assert resumed_optimizer.param_groups[1]["weight_decay"] == 0.0
