import torch
from torch.utils.data.sampler import WeightedRandomSampler
from collections import deque, defaultdict


class OHEMSampler(WeightedRandomSampler):
    """ Weighted Random Sampler with Online Hard Example Mining
    """
    def __init__(self, weights, num_samples=None, window_size=6, scale_factor=10.):
        num_samples = num_samples or weights.shape[0]
        super().__init__(weights, num_samples=num_samples, replacement=True)
        self.base_weights = weights.clone()
        self.losses = defaultdict(lambda: deque(maxlen=window_size))
        self.loss_sma = torch.full((weights.shape[0],), fill_value=float("inf"))
        self.scale_factor = scale_factor

    def update_loss(self, i, loss):
        assert isinstance(loss, (int, float))  # not torch tensor
        self.losses[i].extend([loss])
        self.loss_sma[i] = sum(self.losses[i]) / len(self.losses[i])

    def update_losses(self, indexes, loss):
        # approximate loss for fast batch processing
        # maybe does not work well with large mini batch and few epochs
        for i in indexes:
            self.update_loss(int(i), loss)

    def update_weights(self):
        # scaling weight
        valid_losses = self.loss_sma[self.loss_sma < float("inf")]
        mean = valid_losses.mean()
        stdv = valid_losses.std() + min(abs(mean * 1e-6), 1e-6)

        sigma = 2
        scale_factor = torch.clamp((self.loss_sma - mean) / stdv, -sigma, sigma)
        scale_factor = (scale_factor + sigma) / (2 * sigma)
        scale_factor = torch.clamp(scale_factor * self.scale_factor + 1, 1, self.scale_factor)

        # override weights
        self.weights.copy_(self.base_weights * scale_factor)
        self.weights.div_(self.scale_factor)  # normalize for readability


def _test_ohem():
    weights = torch.ones(8)
    sampler = OHEMSampler(weights)

    print(sampler.weights)
    sampler.update_loss(1, 1)
    sampler.update_loss(2, 2)
    sampler.update_loss(3, 3)
    sampler.update_loss(4, 4)
    sampler.update_losses([1, 2], 1)
    sampler.update_losses(torch.tensor([1, 2]), 1)
    sampler.update_weights()
    print(sampler.weights)
    print(sampler.losses)


if __name__ == "__main__":
    _test_ohem()
