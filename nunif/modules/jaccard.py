from torch import nn


def single_channel_jaccard_index(input, target, threshold=0.5):
    # shape = HW
    assert (input.ndim == 2 and target.ndim == 2 and input.shape == target.shape)

    a = (input >= threshold).long()
    b = (target >= threshold).long()
    a_count = a.sum().item()
    b_count = b.sum().item()
    a_and_b = a.mul(b).sum().item()
    ab = (a_count + b_count - a_and_b)
    if ab > 0.0:
        score = (a_and_b / ab)
    else:
        score = 1.0
    return score


def multi_channel_jaccard_index(input, target, threshold=0.5):
    # shape = CHW
    assert (input.ndim == 3 and target.ndim == 3 and input.shape == target.shape)
    return sum([single_channel_jaccard_index(input[i, :, :], target[i, :, :], threshold)
                for i in range(input.shape[0])]) / input.shape[0]


def batch_jaccard_index(input, target, threshold=0.5):
    # shape = BCWH
    assert (input.ndim == 4 and target.ndim == 4 and input.shape == target.shape)
    return sum([multi_channel_jaccard_index(input[i, :, :, :], target[i, :, :, :], threshold)
                for i in range(input.shape[0])]) / input.shape[0]


def jaccard_index(input, target, threshold=0.5):
    if input.ndim == 4:
        return batch_jaccard_index(input, target, threshold)
    elif input.ndim == 3:
        return multi_channel_jaccard_index(input, target, threshold)
    elif input.ndim == 2:
        return single_channel_jaccard_index(input, target, threshold)
    else:
        raise ValueError("input.ndim not in {4,3,2}")


class JaccardIndex(nn.Module):
    """ Binary Jaccard Index, aka IoU
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, input, target):
        assert (input.shape == target.shape)
        return jaccard_index(input, target, self.threshold)


if __name__ == "__main__":
    import torch

    for shape in ((4, 4), (3, 4, 4), (2, 3, 4, 4)):
        y = torch.rand(shape, dtype=torch.float)
        assert (jaccard_index(y, y) == 1)
        assert (jaccard_index(y, 1 - y) == 0)
