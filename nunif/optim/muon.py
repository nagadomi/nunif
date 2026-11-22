from collections.abc import Callable
from typing import Any, overload

import torch
from torch.optim import Optimizer

_SYNC_KEYS = (
    "lr",
    "weight_decay",
    "momentum",
    "nesterov",
    "ns_coefficients",
    "eps",
    "ns_steps",
    "adjust_lr_fn",
)


def _as_2d_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim >= 3 and all(s == 1 for s in tensor.shape[2:]):
        idx = (slice(None), slice(None)) + (0,) * (tensor.ndim - 2)
        return tensor[idx]
    raise ValueError(
        "MuonWrapper only supports 2D parameters and pointwise conv weights with trailing unit dimensions, "
        f"got shape {tuple(tensor.shape)}"
    )


class MuonWrapper(Optimizer):
    """Thin wrapper around ``torch.optim.Muon`` that also accepts pointwise conv weights.

    Outer ``param_groups`` keep the original ``nn.Parameter`` objects (2D or
    ``(out_c, in_c, 1, ...)``) so ``GradScaler``, ``clip_grad_norm_``,
    ``zero_grad``, and LR/WD schedulers operate transparently. Pointwise conv
    weights are forwarded to ``torch.optim.Muon`` via zero-copy 2D views.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch version")

        defaults: dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        self._initialized = False
        super().__init__(params, defaults)
        self._initialized = True

        self._view_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        inner_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            inner_params: list[torch.Tensor] = []
            for p in group["params"]:
                if p.ndim == 2:
                    inner_params.append(p)
                else:
                    view_2d = _as_2d_view(p.detach())
                    self._view_pairs.append((p, view_2d))
                    inner_params.append(view_2d)
            inner_group = {k: v for k, v in group.items() if k != "params"}
            inner_group["params"] = inner_params
            inner_groups.append(inner_group)

        self._muon: Optimizer = muon_cls(
            inner_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._initialized:
            raise NotImplementedError("MuonWrapper does not support add_param_group() after initialization")
        super().add_param_group(param_group)

    def _sync_inner_groups(self) -> None:
        for outer_group, inner_group in zip(self.param_groups, self._muon.param_groups, strict=True):
            for key in _SYNC_KEYS:
                if key in outer_group:
                    inner_group[key] = outer_group[key]

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sync_inner_groups()
        for orig_param, view_2d in self._view_pairs:
            if orig_param.data_ptr() != view_2d.data_ptr():
                raise RuntimeError("Pointwise conv parameter storage changed after MuonWrapper initialization")
            view_2d.grad = None if orig_param.grad is None else _as_2d_view(orig_param.grad)

        try:
            self._muon.step()
        finally:
            for _, view_2d in self._view_pairs:
                view_2d.grad = None

        return loss

    def state_dict(self) -> dict[str, Any]:
        outer = super().state_dict()
        inner = self._muon.state_dict()
        return {
            "state": inner["state"],
            "param_groups": outer["param_groups"],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
        self._muon.load_state_dict(state_dict)
        self._sync_inner_groups()
