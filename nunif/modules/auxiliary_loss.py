import torch
import torch.nn as nn
from . functional import auxiliary_loss


class AuxiliaryLoss(nn.Module):
    def __init__(self, loss_modules, loss_weights=None):
        super(AuxiliaryLoss, self).__init__()
        if loss_weights is not None:
            loss_weights = torch.ones(len(loss_modules)).float()
        assert (len(loss_modules) == len(loss_modules))
        self.loss_modules = nn.ModuleList(loss_modules)
        self.loss_weights = loss_weights

    def forward(self, inputs, targets):
        return auxiliary_loss(inputs, targets, self.loss_modules, self.loss_weights)
