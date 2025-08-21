import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_loss(real, fake, generator_loss, discriminator_loss, loss_weights):
    loss_fn = generator_loss if fake is None else discriminator_loss
    if isinstance(real, (list, tuple)):
        loss_weights = loss_weights[:len(real)]
        sum_weight = sum(loss_weights)
        loss_weights = [w / sum_weight for w in loss_weights]
        if fake is None:
            fake = (None,) * len(real)
        loss = 0
        for w, r, f in zip(loss_weights, real, fake):
            loss = loss + loss_fn(r, f) * w
        return loss
    else:
        return loss_fn(real, fake)


class GANBCELoss(nn.Module):
    def __init__(self, loss_weights=(1.0,)):
        super().__init__()
        self.loss_weights = loss_weights

    @staticmethod
    def generator_loss(real, fake):
        label_real = torch.ones_like(real)
        return F.binary_cross_entropy_with_logits(real, label_real)

    @staticmethod
    def discriminator_loss(real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        return (F.binary_cross_entropy_with_logits(real, label_real) +
                F.binary_cross_entropy_with_logits(fake, label_fake)) * 0.5

    def forward(self, real, fake=None):
        return _compute_loss(
            real, fake,
            generator_loss=self.generator_loss,
            discriminator_loss=self.discriminator_loss,
            loss_weights=self.loss_weights)


class GANHingeLoss(nn.Module):
    def __init__(self, loss_weights=(1.0,)):
        super().__init__()
        self.loss_weights = loss_weights

    @staticmethod
    def generator_loss(real, fake):
        return -torch.mean(real)

    @staticmethod
    def discriminator_loss(real, fake):
        # real: min: 1, max: -inf
        # fake: min: -1, max: inf
        return (F.relu(1. - real).mean() + F.relu(1. + fake).mean()) * 0.5

    def forward(self, real, fake=None):
        return _compute_loss(
            real, fake,
            generator_loss=self.generator_loss,
            discriminator_loss=self.discriminator_loss,
            loss_weights=self.loss_weights)


class GANHingeClampLoss(GANHingeLoss):
    @staticmethod
    def generator_loss(real, fake):
        # Soft clamp generator loss less than 0
        return F.leaky_relu(1. - real, 0.01).mean()


class GANSoftplusLoss(nn.Module):
    # From SNGAN
    # https://github.com/pfnet-research/sngan_projection/issues/18#issuecomment-392683263
    def __init__(self, loss_weights=(1.0,)):
        super().__init__()
        self.loss_weights = loss_weights

    @staticmethod
    def generator_loss(real, fake):
        return F.softplus(-real).mean()

    @staticmethod
    def discriminator_loss(real, fake):
        return (F.softplus(-real).mean() + F.softplus(fake).mean()) * 0.5

    def forward(self, real, fake=None):
        return _compute_loss(
            real, fake,
            generator_loss=self.generator_loss,
            discriminator_loss=self.discriminator_loss,
            loss_weights=self.loss_weights)
