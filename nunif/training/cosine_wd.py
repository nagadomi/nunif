from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestartsWithFixedWeightDecay(CosineAnnealingWarmRestarts):
    # pytorch adamw weight decay multiplied by lr
    # `param.mul_(1 - lr * weight_decay)`
    # change this to:
    # ```
    # param.mul_(1 - weight_decay)
    # ```
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0) > 0 and "initial_weight_decay" not in group:
                group["_initial_weight_decay"] = group["weight_decay"]
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        super().step(epoch=epoch)
        for group in self.optimizer.param_groups:
            initial_weight_decay = group.get("_initial_weight_decay", 0)
            if initial_weight_decay == 0:
                # no decay param group
                continue
            lr = group["lr"]
            if lr > 0:
                group["weight_decay"] = min(initial_weight_decay / lr, 65000.0)

            # print("update wd", group["weight_decay"], group["weight_decay"] * lr)


class CosineAnnealingWarmRestartsWithScheduledWeightDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1,
                 weight_decay_min=0.001, weight_decay_max=0.05):
        self.weight_decay_min = weight_decay_min
        self.weight_decay_max = weight_decay_max
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        super().step(epoch=epoch)
        for group in self.optimizer.param_groups:
            if group.get("weight_decay", 0) == 0:
                continue

            initial_lr = group["initial_lr"]
            lr = group["lr"]
            weight_decay_factor = 1.0 - max(lr - self.eta_min, 0) / initial_lr
            weight_decay = self.weight_decay_min
            weight_decay += (self.weight_decay_max - self.weight_decay_min) * weight_decay_factor
            # print("update wd", group["weight_decay"], weight_decay)
            group["weight_decay"] = weight_decay
