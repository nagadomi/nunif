from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, overload

import torch
from torch.optim import Optimizer


class CompositeOptimizer(Optimizer):
    """Wraps multiple PyTorch optimizers as a single Optimizer instance.

    Shares child ``param_groups`` dict references directly so LR/WD schedulers
    and ``GradScaler`` operate across all child optimizers seamlessly.
    """

    def __init__(self, optimizers: Sequence[Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer")
        self.optimizers: list[Optimizer] = list(optimizers)

        seen_param_ids: set[int] = set()
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    param_id = id(p)
                    if param_id in seen_param_ids:
                        raise ValueError("Duplicate parameter found across child optimizers")
                    seen_param_ids.add(param_id)

        # Initialize Optimizer internals without copying param_group dicts so
        # schedulers and GradScaler mutate the dicts owned by child optimizers.
        self.defaults: dict[str, Any] = {}
        self._optimizer_step_pre_hooks: OrderedDict[int, Any] = OrderedDict()
        self._optimizer_step_post_hooks: OrderedDict[int, Any] = OrderedDict()
        self._optimizer_state_dict_pre_hooks: OrderedDict[int, Any] = OrderedDict()
        self._optimizer_state_dict_post_hooks: OrderedDict[int, Any] = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks: OrderedDict[int, Any] = OrderedDict()
        self._optimizer_load_state_dict_post_hooks: OrderedDict[int, Any] = OrderedDict()
        self._patch_step_function()
        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []
        self._warned_capturable_if_run_uncaptured = True
        self._sync_param_groups()

    def _sync_param_groups(self) -> None:
        self.param_groups[:] = [group for opt in self.optimizers for group in opt.param_groups]

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        raise NotImplementedError("CompositeOptimizer does not support add_param_group()")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        opt_states = state_dict.get("optimizers")
        if not isinstance(opt_states, list) or len(opt_states) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer state dicts, "
                f"got {len(opt_states) if isinstance(opt_states, list) else type(opt_states).__name__}"
            )
        for opt, opt_state in zip(self.optimizers, opt_states, strict=True):
            opt.load_state_dict(opt_state)
        self._sync_param_groups()

    def train(self) -> None:
        for opt in self.optimizers:
            train_fn = getattr(opt, "train", None)
            if callable(train_fn):
                train_fn()

    def eval(self) -> None:
        for opt in self.optimizers:
            eval_fn = getattr(opt, "eval", None)
            if callable(eval_fn):
                eval_fn()
