import torch
import torch.nn as nn


class DiscriminatorBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, is_real):
        if is_real:
            label = torch.ones(input.shape, dtype=input.dtype, device=input.device)
        else:
            label = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
        return self.bce(input, label)
