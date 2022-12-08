from torch import nn


class JaccardIndex(nn.Module):
    """ aka IoU
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, input, target, batch=True):
        assert(input.shape == target.shape)
        score = 0.0
        count = 0.0
        if not batch:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        for y, t in zip(input, target):
            a = (y >= self.threshold).long()
            b = (t >= self.threshold).long()
            a_count = a.sum().item()
            b_count = b.sum().item()
            a_and_b = a.mul(b).sum().item()
            ab = (a_count + b_count - a_and_b)
            if ab > 0.0:
                score += (a_and_b / ab)
            else:
                score += 1.0
            count += 1.0
        return score / count
