import torch


def inplace_clip(x, min_value, max_value):
    return torch.clamp_(x, min_value, max_value)


def weighted_huber_loss(input, target, weight, gamma=1.0, reduction='mean'):
    t = torch.abs(input - target).mul_(weight)
    loss = torch.where(t < gamma, 0.5 * t **2, (t - 0.5 * gamma) * gamma)
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == 'spatial_mean':
        bs, ch, h, w = input.shape
        loss = loss.view(bs, ch, -1).mean(dim=2).sum(dim=1).mean()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"undefined reduction: {reduction}")
    return loss

def auxiliary_loss(inputs, targets, modules, weights):
    assert(len(inputs) == len(targets) == len(modules) == len(weights))
    n = len(inputs)
    loss = None
    for i in range(n):
        z = modules[i].forward(inputs[i], targets[i]) * weights[i]
        if loss is None:
            loss = z
        else:
            loss = loss + z
    return loss
