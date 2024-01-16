import torch
import torch.nn as nn


def auxiliary_loss(inputs, targets, modules, weights):
    assert (len(inputs) == len(targets) and len(modules) == len(weights))
    return sum([modules[i].forward(inputs[i], targets[i]) * weights[i] for i in range(len(inputs))])


class AuxiliaryLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(AuxiliaryLoss, self).__init__()
        if weight is None:
            weight = torch.tensor([1.0 / len(losses)] * len(losses), dtype=torch.float)
        if isinstance(weight, (tuple, list)):
            weight = torch.tensor(weight, dtype=torch.float)

        assert (len(losses) == len(weight))
        self.losses = nn.ModuleList(losses)
        self.weight = weight

    def forward(self, inputs, targets):
        if isinstance(inputs, (list, tuple)):
            if not isinstance(targets, (list, tuple)):
                targets = [targets] * len(inputs)
            return auxiliary_loss(inputs, targets, self.losses, self.weight)
        else:
            return self.losses[0](inputs, targets)
