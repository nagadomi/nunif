import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim.lr_scheduler import ChainedScheduler, ConstantLR, CosineAnnealingWarmRestarts, StepLR
from torchvision.models.swin_transformer import ShiftedWindowAttention, ShiftedWindowAttentionV2

from nunif.modules.fusion import AdaptiveWeight, Lerp
from nunif.modules.norm import GroupNormNoBias, LayerNormNoBias2d, ReparamBatchNorm2d, RMSNorm1, RMSNorm2d
from nunif.optim import CompositeOptimizer, MuonWrapper
from nunif.training.cosine_wd import (
    CosineAnnealingWarmRestartsWithFixedWeightDecay,
    CosineAnnealingWarmRestartsWithScheduledWeightDecay,
)
from nunif.training.trainer import Trainer, create_trainer_default_parser
from nunif.training.weight_decay_config import configure_muon, split_muon_params


def _param_ids(params: list[nn.Parameter]) -> set[int]:
    return {id(p) for p in params}


def test_split_muon_standard_and_custom_modules() -> None:
    ln_nobias2d = LayerNormNoBias2d(4)
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
            "rms1": RMSNorm1(4),
            "rms2d": RMSNorm2d(4),
            "ln": nn.LayerNorm(4),
            "bn2d": nn.BatchNorm2d(4),
            "gn": nn.GroupNorm(2, 4),
            "in2d": nn.InstanceNorm2d(4, affine=True),
            "rmsnorm": nn.RMSNorm(4),
            "emb": nn.Embedding(8, 4),
            "emb_bag": nn.EmbeddingBag(8, 4),
            "prelu": nn.PReLU(4),
            "ln_nobias2d": ln_nobias2d,
            "gn_nobias": GroupNormNoBias(2, 4),
            "reparam_bn": ReparamBatchNorm2d(4),
            "lerp": Lerp(),
            "adaptive_weight": AdaptiveWeight(3),
        }
    )

    muon_decay, adamw_decay, adamw_no_decay = split_muon_params(model)
    muon_ids = _param_ids(muon_decay)
    adamw_decay_ids = _param_ids(adamw_decay)
    adamw_no_decay_ids = _param_ids(adamw_no_decay)

    assert id(model["linear"].weight) in muon_ids
    assert id(model["linear"].bias) in adamw_no_decay_ids

    for key in ("bilinear", "conv1d", "conv2d", "conv3d", "convt1d", "convt2d", "convt3d"):
        mod = model[key]
        assert id(mod.weight) in adamw_decay_ids
        assert id(mod.bias) in adamw_no_decay_ids

    assert id(model["rms1"].weight) in adamw_decay_ids
    assert id(model["rms2d"].weight) in adamw_decay_ids

    for key in (
        "ln",
        "bn2d",
        "gn",
        "in2d",
        "rmsnorm",
        "emb",
        "emb_bag",
        "prelu",
        "gn_nobias",
        "reparam_bn",
        "lerp",
        "adaptive_weight",
    ):
        mod = model[key]
        assert id(mod.weight) in adamw_no_decay_ids
        if getattr(mod, "bias", None) is not None:
            assert id(mod.bias) in adamw_no_decay_ids

    assert id(ln_nobias2d.norm.weight) in adamw_no_decay_ids
    assert not (muon_ids & adamw_decay_ids)
    assert not (muon_ids & adamw_no_decay_ids)
    assert not (adamw_decay_ids & adamw_no_decay_ids)


def test_split_muon_attention_rnn_and_spectral_norm() -> None:
    mha_same = nn.MultiheadAttention(embed_dim=8, num_heads=2, add_bias_kv=True)
    mha_diff = nn.MultiheadAttention(embed_dim=8, num_heads=2, kdim=4, vdim=4)
    swa = ShiftedWindowAttention(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    swa_v2 = ShiftedWindowAttentionV2(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    lstm = nn.LSTM(4, 4, num_layers=2)
    gru_cell = nn.GRUCell(4, 4)
    sn_linear = spectral_norm(nn.Linear(4, 4))

    model = nn.ModuleDict(
        {
            "mha_same": mha_same,
            "mha_diff": mha_diff,
            "swa": swa,
            "swa_v2": swa_v2,
            "lstm": lstm,
            "gru_cell": gru_cell,
            "sn_linear": sn_linear,
        }
    )

    muon_decay, adamw_decay, adamw_no_decay = split_muon_params(model)
    muon_ids = _param_ids(muon_decay)
    adamw_decay_ids = _param_ids(adamw_decay)
    adamw_no_decay_ids = _param_ids(adamw_no_decay)

    assert len(adamw_decay_ids) == 0

    assert id(mha_same.in_proj_weight) in muon_ids
    assert id(mha_same.out_proj.weight) in muon_ids
    assert id(mha_same.in_proj_bias) in adamw_no_decay_ids
    assert id(mha_same.bias_k) in adamw_no_decay_ids
    assert id(mha_same.bias_v) in adamw_no_decay_ids
    assert id(mha_diff.q_proj_weight) in muon_ids
    assert id(mha_diff.k_proj_weight) in muon_ids
    assert id(mha_diff.v_proj_weight) in muon_ids
    assert id(mha_diff.out_proj.weight) in muon_ids

    assert id(swa.qkv.weight) in muon_ids
    assert id(swa.proj.weight) in muon_ids
    assert id(swa.relative_position_bias_table) in adamw_no_decay_ids
    assert id(swa_v2.logit_scale) in adamw_no_decay_ids

    for name, param in lstm.named_parameters():
        if name.startswith("weight_"):
            assert id(param) in muon_ids
        else:
            assert id(param) in adamw_no_decay_ids
    for name, param in gru_cell.named_parameters():
        if name.startswith("weight_"):
            assert id(param) in muon_ids
        else:
            assert id(param) in adamw_no_decay_ids

    for param in sn_linear.parameters():
        assert id(param) in adamw_no_decay_ids


def test_split_muon_weight_decay_hook_frozen_tied_and_dataparallel() -> None:
    class NoDecayLinear(nn.Linear):
        def weight_decay_config(self, pn: str) -> bool | None:
            if pn == "weight":
                return False
            return None

    class SampleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.z_fc = nn.Linear(4, 4)
            self.a_fc = nn.Linear(4, 4)
            self.tied_fc = nn.Linear(4, 4, bias=False)
            self.tied_fc.weight = self.a_fc.weight
            self.no_decay_fc = NoDecayLinear(4, 4)
            self.frozen_fc = nn.Linear(4, 4)
            self.frozen_fc.requires_grad_(False)

    base = SampleModel()
    wrapped = nn.DataParallel(base)
    muon_decay, adamw_decay, adamw_no_decay = split_muon_params(wrapped)

    assert muon_decay == [base.a_fc.weight, base.z_fc.weight]
    assert adamw_decay == []
    assert _param_ids(adamw_no_decay) == {
        id(base.a_fc.bias),
        id(base.z_fc.bias),
        id(base.no_decay_fc.weight),
        id(base.no_decay_fc.bias),
    }

    class ConflictingTiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(4, 4)
            self.head = nn.Linear(4, 4, bias=False)
            self.head.weight = self.emb.weight

    with pytest.raises(ValueError, match="Conflicting optimizer decisions"):
        split_muon_params(ConflictingTiedModel())


def test_configure_muon_step_and_schedulers_and_resume() -> None:
    class MixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)
            self.norm = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x).mean(dim=(2, 3))
            return self.fc(self.norm(h))

    torch.manual_seed(0)
    model = MixedModel()
    opt = configure_muon(model, lr=0.01, weight_decay=0.001)
    assert isinstance(opt, CompositeOptimizer)
    assert len(opt.optimizers) == 2
    assert isinstance(opt.optimizers[0], MuonWrapper)
    assert isinstance(opt.optimizers[1], torch.optim.AdamW)
    # 3 groups: muon decay, adamw decay, adamw no_decay
    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]["weight_decay"] == 0.001
    assert opt.param_groups[1]["weight_decay"] == 0.001
    assert opt.param_groups[2]["weight_decay"] == 0.0

    fc_before = model.fc.weight.detach().clone()
    conv_before = model.conv.weight.detach().clone()
    bias_before = model.fc.bias.detach().clone()

    x = torch.randn(2, 4, 4, 4)
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert not torch.equal(model.fc.weight, fc_before)
    assert not torch.equal(model.conv.weight, conv_before)
    assert not torch.equal(model.fc.bias, bias_before)
    assert model.fc.weight.grad is None
    assert model.conv.weight.grad is None

    # Verify CosineAnnealingWarmRestartsWithFixedWeightDecay and resume
    scheduler = CosineAnnealingWarmRestartsWithFixedWeightDecay(opt, T_0=10, eta_min=0.001)
    assert opt.param_groups[0]["_initial_weight_decay"] == 0.001
    assert opt.param_groups[1]["_initial_weight_decay"] == 0.001
    assert "_initial_weight_decay" not in opt.param_groups[2]

    scheduler.step()
    assert opt.param_groups[0]["lr"] == opt.optimizers[0].param_groups[0]["lr"]
    assert opt.param_groups[1]["lr"] == opt.optimizers[1].param_groups[0]["lr"]
    assert opt.param_groups[2]["weight_decay"] == 0.0

    opt_state = opt.state_dict()
    sched_state = scheduler.state_dict()

    resumed_opt = configure_muon(model, lr=0.01, weight_decay=0.001)
    resumed_sched = CosineAnnealingWarmRestartsWithFixedWeightDecay(resumed_opt, T_0=10, eta_min=0.001)
    resumed_opt.load_state_dict(opt_state)
    resumed_sched.load_state_dict(sched_state)

    assert resumed_opt.param_groups[0] is resumed_opt.optimizers[0].param_groups[0]
    assert resumed_opt.param_groups[1] is resumed_opt.optimizers[1].param_groups[0]
    assert resumed_opt.param_groups[2] is resumed_opt.optimizers[1].param_groups[1]
    assert resumed_opt.param_groups[0]["_initial_weight_decay"] == 0.001

    resumed_sched.step()
    assert resumed_opt.param_groups[0]["lr"] == resumed_opt.optimizers[0].param_groups[0]["lr"]
    assert resumed_opt.param_groups[2]["weight_decay"] == 0.0

    # Verify step after resume works cleanly
    model(x).pow(2).mean().backward()
    resumed_opt.step()
    resumed_opt.zero_grad()


def test_configure_muon_other_schedulers() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

    opt1 = configure_muon(model, lr=0.01, weight_decay=0.01)
    sched1 = ChainedScheduler(
        [
            ConstantLR(opt1, factor=0.1, total_iters=2),
            StepLR(opt1, step_size=1, gamma=0.5),
        ]
    )
    opt1.step()
    sched1.step()
    assert opt1.param_groups[0]["lr"] == opt1.optimizers[0].param_groups[0]["lr"]
    assert opt1.param_groups[1]["lr"] == opt1.optimizers[1].param_groups[0]["lr"]

    opt2 = configure_muon(model, lr=0.01, weight_decay=0.01)
    sched2 = CosineAnnealingWarmRestartsWithScheduledWeightDecay(
        opt2, T_0=10, eta_min=0.001, weight_decay_min=0.01, weight_decay_max=0.05
    )
    opt2.step()
    sched2.step()
    assert opt2.param_groups[0]["weight_decay"] > 0.01
    assert opt2.param_groups[1]["weight_decay"] == 0.0

    opt3 = configure_muon(model, lr=0.01, weight_decay=0.01)
    sched3 = CosineAnnealingWarmRestarts(opt3, T_0=10, eta_min=0.001)
    opt3.step()
    sched3.step()
    assert opt3.param_groups[0]["lr"] == opt3.optimizers[0].param_groups[0]["lr"]


def test_configure_muon_edge_cases_and_composite_validation() -> None:
    # Pure CNN: falls back to AdamW only
    cnn = nn.Conv2d(4, 4, 3)
    cnn_opt = configure_muon(cnn, lr=0.01, weight_decay=0.01)
    assert isinstance(cnn_opt, CompositeOptimizer)
    assert len(cnn_opt.optimizers) == 1
    assert isinstance(cnn_opt.optimizers[0], torch.optim.AdamW)

    # Bias-free Linear: Muon only
    linear_only = nn.Linear(4, 4, bias=False)
    muon_only_opt = configure_muon(linear_only, lr=0.01, weight_decay=0.01)
    assert isinstance(muon_only_opt, CompositeOptimizer)
    assert len(muon_only_opt.optimizers) == 1
    assert isinstance(muon_only_opt.optimizers[0], MuonWrapper)

    # All frozen: raises ValueError
    frozen = nn.Linear(4, 4)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="empty parameter list"):
        configure_muon(frozen)

    # Duplicate params across child optimizers: raises ValueError
    p = nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="Duplicate parameter"):
        CompositeOptimizer([torch.optim.Muon([p], lr=0.01), torch.optim.AdamW([p], lr=0.01)])

    with pytest.raises(ValueError, match="at least one optimizer"):
        CompositeOptimizer([])


def test_configure_muon_with_grad_scaler() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 4)).to(device)
    opt = configure_muon(model, lr=0.01, weight_decay=0.01, fused=True if use_amp else None)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    x = torch.randn(2, 8, device=device)
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        loss = model(x).pow(2).mean()

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    for group in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0, error_if_nonfinite=False)
    scaler.step(opt)
    opt.zero_grad()
    scaler.update()


def test_trainer_create_optimizer_muon() -> None:
    class DummyTrainer(Trainer):
        def create_dataloader(self, type: str) -> None:
            return None

        def create_env(self) -> None:
            return None

    parser = create_trainer_default_parser()
    args = parser.parse_args(
        [
            "--data-dir",
            "dummy_data",
            "--model-dir",
            "dummy_model",
            "--optimizer",
            "muon",
            "--learning-rate",
            "0.002",
            "--weight-decay",
            "0.01",
            "--muon-momentum",
            "0.92",
            "--muon-adjust-lr-fn",
            "original",
        ]
    )
    trainer = DummyTrainer(args)
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    opt = trainer.create_optimizer(model)
    assert isinstance(opt, CompositeOptimizer)
    assert isinstance(opt.optimizers[0], MuonWrapper)
    assert isinstance(opt.optimizers[1], torch.optim.AdamW)
    assert opt.optimizers[0].param_groups[0]["lr"] == 0.002
    assert opt.optimizers[0].param_groups[0]["weight_decay"] == 0.01
    assert opt.optimizers[0].param_groups[0]["momentum"] == 0.92
    assert opt.optimizers[0].param_groups[0]["adjust_lr_fn"] == "original"
    assert opt.optimizers[1].param_groups[0]["lr"] == 0.002
    assert opt.optimizers[1].param_groups[0]["weight_decay"] == 0.0


def test_split_muon_pointwise_convs() -> None:
    model = nn.ModuleDict(
        {
            "conv1d_1x1": nn.Conv1d(4, 8, kernel_size=1),
            "conv2d_1x1": nn.Conv2d(4, 8, kernel_size=1),
            "conv3d_1x1": nn.Conv3d(4, 8, kernel_size=1),
            "conv1d_grouped": nn.Conv1d(4, 8, kernel_size=1, groups=2),
            "conv2d_grouped": nn.Conv2d(4, 4, kernel_size=1, groups=4),
            "conv3d_grouped": nn.Conv3d(4, 8, kernel_size=1, groups=2),
            "conv2d_3x3": nn.Conv2d(4, 8, kernel_size=3),
            "convt2d_1x1": nn.ConvTranspose2d(4, 8, kernel_size=1),
            "sn_conv2d_1x1": spectral_norm(nn.Conv2d(4, 8, kernel_size=1)),
        }
    )

    muon_decay, adamw_decay, adamw_no_decay = split_muon_params(model)
    muon_ids = _param_ids(muon_decay)
    adamw_decay_ids = _param_ids(adamw_decay)
    adamw_no_decay_ids = _param_ids(adamw_no_decay)

    for key in ("conv1d_1x1", "conv2d_1x1", "conv3d_1x1"):
        mod = model[key]
        assert id(mod.weight) in muon_ids
        assert id(mod.bias) in adamw_no_decay_ids

    for key in ("conv1d_grouped", "conv2d_grouped", "conv3d_grouped", "conv2d_3x3", "convt2d_1x1"):
        mod = model[key]
        assert id(mod.weight) in adamw_decay_ids
        assert id(mod.bias) in adamw_no_decay_ids

    for param in model["sn_conv2d_1x1"].parameters():
        assert id(param) in adamw_no_decay_ids


def test_muon_wrapper_pointwise_convs_step_channels_last_resume_and_amp() -> None:
    class PointwiseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv1d(4, 6, kernel_size=1)
            self.c2 = nn.Conv2d(4, 6, kernel_size=1)
            self.c3 = nn.Conv3d(4, 6, kernel_size=1)
            self.fc = nn.Linear(6, 4)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
            y1 = self.c1(x1).mean(dim=-1)
            y2 = self.c2(x2).mean(dim=(2, 3))
            y3 = self.c3(x3).mean(dim=(2, 3, 4))
            return self.fc(y1 + y2 + y3)

    torch.manual_seed(0)
    model = PointwiseModel()
    model.c2.weight.data = model.c2.weight.data.to(memory_format=torch.channels_last)

    opt = configure_muon(model, lr=0.01, weight_decay=0.001)
    sched = CosineAnnealingWarmRestartsWithFixedWeightDecay(opt, T_0=10, eta_min=0.001)

    before = {
        "c1": model.c1.weight.detach().clone(),
        "c2": model.c2.weight.detach().clone(),
        "c3": model.c3.weight.detach().clone(),
        "fc": model.fc.weight.detach().clone(),
    }

    x1 = torch.randn(2, 4, 5)
    x2 = torch.randn(2, 4, 4, 4)
    x3 = torch.randn(2, 4, 2, 3, 3)

    model(x1, x2, x3).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()
    sched.step()

    assert not torch.equal(model.c1.weight, before["c1"])
    assert not torch.equal(model.c2.weight, before["c2"])
    assert not torch.equal(model.c3.weight, before["c3"])
    assert not torch.equal(model.fc.weight, before["fc"])
    assert model.c2.weight.is_contiguous(memory_format=torch.channels_last)
    assert model.c1.weight.grad is None
    assert model.c2.weight.grad is None
    assert model.c3.weight.grad is None

    # Verify resume preserves momentum buffers and produces identical next step
    opt_state = copy.deepcopy(opt.state_dict())
    sched_state = copy.deepcopy(sched.state_dict())
    model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Step 2 on original
    model(x1, x2, x3).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()
    expected_c2 = model.c2.weight.detach().clone()

    # Step 2 on resumed
    model2 = PointwiseModel()
    model2.load_state_dict(model_state)
    opt2 = configure_muon(model2, lr=0.01, weight_decay=0.001)
    sched2 = CosineAnnealingWarmRestartsWithFixedWeightDecay(opt2, T_0=10, eta_min=0.001)
    opt2.load_state_dict(opt_state)
    sched2.load_state_dict(sched_state)

    model2(x1, x2, x3).pow(2).mean().backward()
    opt2.step()
    opt2.zero_grad()
    assert torch.allclose(model2.c2.weight, expected_c2)

    # Verify non-pointwise 4D weight raises ValueError in MuonWrapper
    bad_conv = nn.Conv2d(4, 4, kernel_size=3)
    with pytest.raises(ValueError, match="pointwise conv"):
        MuonWrapper([bad_conv.weight], lr=0.01)

    # Verify GradScaler with pointwise convs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_model = PointwiseModel().to(device)
    amp_opt = configure_muon(amp_model, lr=0.01, weight_decay=0.01, fused=True if use_amp else None)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        loss = amp_model(x1.to(device), x2.to(device), x3.to(device)).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(amp_opt)
    for group in amp_opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0, error_if_nonfinite=False)
    scaler.step(amp_opt)
    amp_opt.zero_grad()
    scaler.update()


def test_muon_config_hook_ancestor_resolution_and_validation() -> None:
    class SubBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keep_muon = nn.Conv2d(4, 4, 1, 1)
            self.other = nn.Linear(4, 4)

        def muon_config(self, pn: str) -> bool | None:
            if pn == "keep_muon.weight":
                return True
            return None

    class ModelWithOutputHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Conv2d(4, 4, 1, 1)
            self.to_image = nn.Conv2d(4, 3, 1, 1)
            self.sub = SubBlock()
            self.custom_mat = nn.Parameter(torch.randn(4, 4))

        def weight_decay_config(self, pn: str) -> bool | None:
            if pn == "custom_mat":
                return True
            return None

        def muon_config(self, pn: str) -> bool | None:
            if pn.startswith("to_image."):
                return False
            if pn.startswith("sub."):
                # Inner SubBlock overrides this for sub.keep_muon.weight
                return False
            if pn == "custom_mat":
                return True
            return None

    base = ModelWithOutputHead()
    muon_decay, adamw_decay, adamw_no_decay = split_muon_params(nn.DataParallel(base))
    muon_ids = _param_ids(muon_decay)
    adamw_decay_ids = _param_ids(adamw_decay)
    adamw_no_decay_ids = _param_ids(adamw_no_decay)

    assert id(base.body.weight) in muon_ids
    assert id(base.sub.keep_muon.weight) in muon_ids
    assert id(base.custom_mat) in muon_ids

    # Rejected by muon_config -> stays decay-enabled in AdamW
    assert id(base.to_image.weight) in adamw_decay_ids
    assert id(base.sub.other.weight) in adamw_decay_ids

    assert id(base.body.bias) in adamw_no_decay_ids
    assert id(base.to_image.bias) in adamw_no_decay_ids
    assert id(base.sub.keep_muon.bias) in adamw_no_decay_ids
    assert id(base.sub.other.bias) in adamw_no_decay_ids

    # Incompatible shape forced to True raises ValueError
    class BadShapeMuon(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)

        def muon_config(self, pn: str) -> bool | None:
            if pn == "conv.weight":
                return True
            return None

    with pytest.raises(ValueError, match="incompatible parameter shape"):
        split_muon_params(BadShapeMuon())

    # Non-decay parameter forced to True raises ValueError
    class NonDecayMuon(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(8, 4)

        def muon_config(self, pn: str) -> bool | None:
            if pn == "emb.weight":
                return True
            return None

    with pytest.raises(ValueError, match="non-decay parameter"):
        split_muon_params(NonDecayMuon())

    # Invalid return type raises TypeError
    class InvalidHookType(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def muon_config(self, pn: str) -> str:
            return "no"

    with pytest.raises(TypeError, match="must return bool or None"):
        split_muon_params(InvalidHookType())

    # Tied weight conflict via muon_config raises ValueError
    class TiedConflict(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4, bias=False)
            self.b = nn.Linear(4, 4, bias=False)
            self.b.weight = self.a.weight

        def muon_config(self, pn: str) -> bool | None:
            if pn == "b.weight":
                return False
            return None

    with pytest.raises(ValueError, match="Conflicting optimizer decisions"):
        split_muon_params(TiedConflict())
