import torch
import torch.nn as nn

from .init import basic_module_init


class WindowScoreBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads

        index, unique_delta = _gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)
        if hidden_dim is None:
            hidden_dim = int((self.window_size1[0] * self.window_size1[1]) ** 0.5) * 2
        if self.num_heads is None:
            output_dim = 1
        else:
            output_dim = num_heads

        self.to_bias = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True), nn.GELU(), nn.Linear(hidden_dim, output_dim, bias=True)
        )

        basic_module_init(self)

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        bias = self.to_bias(self.delta)
        bias = bias[self.index]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = bias.reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = bias.permute(1, 0).contiguous().reshape(self.num_heads, N1, N2)
        return bias


@torch.no_grad()
def _gen_window_score_bias_input(window_size1, window_size2, reduction):
    N1 = window_size1[0] * window_size1[1]
    N2 = window_size2[0] * window_size2[1]

    positions1 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size1[0]), torch.arange(0, window_size1[1]), indexing="ij"), dim=2
    ).reshape(N1, 2)

    positions2 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size2[0]), torch.arange(0, window_size2[1]), indexing="ij"), dim=2
    ).reshape(N2, 2)
    positions2.mul_(reduction)

    delta = torch.zeros((N1, N2, 2), dtype=torch.long)
    for i in range(N1):
        for j in range(N2):
            delta[i][j] = positions1[i] - positions2[j]

    delta = delta.view(N1 * N2, 2)
    delta = [tuple(p) for p in delta.tolist()]
    unique_delta = sorted(list(set(delta)))
    index = [unique_delta.index(d) for d in delta]
    index = torch.tensor(index, dtype=torch.int64)
    unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
    unique_delta = unique_delta / unique_delta.abs().max()
    return index, unique_delta


class WindowRelativeScoreBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        self.num_heads = num_heads
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2

        index, _ = _gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index.to(torch.int32))
        if num_heads is None:
            self.bias = nn.Parameter(torch.zeros((index.max() + 1,), dtype=torch.float32))
        else:
            self.bias = nn.Parameter(torch.zeros((num_heads, index.max() + 1), dtype=torch.float32))

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = self.bias[self.index].reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = self.bias[:, self.index].reshape(self.num_heads, N1, N2)

        return bias


class WindowDistanceScoreBias(nn.Module):
    def __init__(self, window_size, max_distance=None, num_heads=None):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)

        mask = None
        if num_heads is not None:
            distance = window_distance_matrix(self.window_size)
            distance = distance.expand(num_heads, *distance.shape)
            distance_bias = (1.0 + distance).log().neg()
            self.register_buffer("distance_bias", distance_bias)
            self.scale_bias = nn.Parameter(torch.zeros((num_heads, 1, 1), dtype=torch.float32))

            if max_distance is not None:
                if isinstance(max_distance, (list, tuple)):
                    if len(max_distance) != num_heads:
                        assert num_heads % len(max_distance) == 0
                        max_distance = max_distance * (num_heads // len(max_distance))
                    max_distance = torch.tensor(max_distance, dtype=torch.float32).view(num_heads, 1, 1)
                    mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))
                else:
                    mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))
        else:
            distance = window_distance_matrix(self.window_size)
            distance_bias = (1.0 + distance).log().neg()
            self.register_buffer("distance_bias", distance_bias)
            self.scale_bias = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

            if max_distance is not None:
                mask = torch.where(distance <= max_distance, torch.zeros_like(distance), -float("inf"))

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self):
        scale = self.scale_bias.exp()
        # print(self.window_size, scale.flatten().tolist())
        bias = self.distance_bias * scale
        if self.mask is not None:
            bias = bias + self.mask

        return bias


@torch.no_grad()
def window_distance_matrix(window_size):
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    else:
        window_size = window_size

    N = window_size[0] * window_size[1]
    positions = torch.stack(
        torch.meshgrid(torch.arange(0, window_size[0]), torch.arange(0, window_size[1]), indexing="ij"), dim=2
    ).reshape(N, 2)

    positions = positions.to(torch.float32)
    distance = torch.cat(
        [((positions[i].view(1, 2) - positions) ** 2).sum(dim=1) ** 0.5 for i in range(positions.shape[0])], dim=0
    )
    distance = distance.view(N, N)
    return distance


@torch.no_grad()
def _gen_window_score_bias_input_3d(window_size1, window_size2, reduction):
    D1, H1, W1 = window_size1
    D2, H2, W2 = window_size2

    # positions1: (N1, 3)
    positions1 = torch.stack(
        torch.meshgrid(torch.arange(0, D1), torch.arange(0, H1), torch.arange(0, W1), indexing="ij"), dim=3
    ).reshape(-1, 3)

    # positions2: (N2, 3)
    positions2 = torch.stack(
        torch.meshgrid(torch.arange(0, D2), torch.arange(0, H2), torch.arange(0, W2), indexing="ij"), dim=3
    ).reshape(-1, 3)
    positions2.mul_(reduction)

    N1 = positions1.shape[0]
    N2 = positions2.shape[0]

    delta = torch.zeros((N1, N2, 3), dtype=torch.long)
    for i in range(N1):
        for j in range(N2):
            delta[i, j] = positions1[i] - positions2[j]

    delta = delta.view(N1 * N2, 3)
    delta = [tuple(p) for p in delta.tolist()]
    unique_delta = sorted(list(set(delta)))
    index = [unique_delta.index(d) for d in delta]
    index = torch.tensor(index, dtype=torch.int64)
    unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
    unique_delta = unique_delta / unique_delta.abs().max()

    # print(len(unique_delta))

    return index, unique_delta


class WindowScoreBias3d(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size1 = [window_size] * 3
        else:
            window_size1 = window_size

        D, H, W = window_size1
        assert D % reduction == 0 and H % reduction == 0 and W % reduction == 0

        window_size2 = [D // reduction, H // reduction, W // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads

        index, unique_delta = _gen_window_score_bias_input_3d(window_size1, window_size2, reduction)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)

        if hidden_dim is None:
            hidden_dim = int((D * H * W) ** 0.5) * 2
            if hidden_dim % 4 != 0:
                hidden_dim = hidden_dim + (4 - hidden_dim % 4)
        if self.num_heads is None:
            output_dim = 1
        else:
            output_dim = num_heads

        self.to_bias = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=True), nn.GELU(), nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1] * self.window_size1[2]
        N2 = self.window_size2[0] * self.window_size2[1] * self.window_size2[2]

        bias = self.to_bias(self.delta)
        bias = bias[self.index]

        if self.num_heads is None:
            bias = bias.reshape(N1, N2)
        else:
            bias = bias.permute(1, 0).contiguous().reshape(self.num_heads, N1, N2)
        return bias


if __name__ == "__main__":
    pass
