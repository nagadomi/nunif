import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, real, fake=None):
        if real is not None and fake is not None:
            t_real = torch.ones(real.shape, dtype=real.dtype, device=real.device)
            t_fake = torch.zeros(real.shape, dtype=real.dtype, device=real.device)
            return (self.bce(real, t_real).mean() + self.bce(fake, t_fake)) * 0.5
        else:
            t_real = torch.ones(real.shape, dtype=real.dtype, device=real.device,
                                requires_grad=False)
            return self.bce(real, t_real).mean()


class DiscriminatorHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake=None):
        if real is not None and fake is not None:
            return (F.relu(1. - real).mean() + F.relu(1. + fake).mean()) * 0.5
        else:
            loss = -torch.mean(real)
            return loss
