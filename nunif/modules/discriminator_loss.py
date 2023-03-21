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
    def __init__(self, loss_weights=(1.0,)):
        super().__init__()
        self.loss_weights = loss_weights

    def forward(self, real, fake=None):
        if real is not None and fake is not None:
            if isinstance(real, (list, tuple)):
                assert len(real) == len(fake) == len(self.loss_weights)
                loss = 0
                for w, r, f in zip(self.loss_weights, real, fake):
                    loss = loss + (F.relu(1. - r).mean() + F.relu(1. + f).mean()) * 0.5 * w
                return loss
            else:
                return (F.relu(1. - real).mean() + F.relu(1. + fake).mean()) * 0.5
        else:
            if isinstance(real, (list, tuple)):
                assert len(real) == len(self.loss_weights)
                loss = 0
                for w, r in zip(self.loss_weights, real):
                    loss = loss + (-torch.mean(r) * w)
                return loss
            else:
                loss = -torch.mean(real)
            return loss
