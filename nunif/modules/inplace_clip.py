import torch


class InplaceClip(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super(InplaceClip, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp_(x, self.min_value, self.max_value)
