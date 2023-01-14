import torch


# for VAE

def gaussian_noise(mean, log_var):
    # NOTE:
    # math.exp(log_var * 0.5) == math.exp(log_var/math.log(math.e**2)) == math.sqrt(math.exp(log_var))
    standard_deviation = torch.exp(log_var * 0.5)
    noise = torch.randn(mean.shape, device=mean.get_device())
    return mean + (noise * standard_deviation)


def gaussian_kl_divergence_loss(mean, log_var, reduction="mean"):
    var = torch.exp(log_var)
    mean2 = mean ** 2
    kl = 0.5 * (1 + log_var - mean2 - var)
    if reduction == "mean":
        kl = kl.mean()
    elif reduction == "sum":
        # assuming shape[0] is batch dim
        kl = kl.sum()
    return -kl
