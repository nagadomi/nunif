import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_eval

from .init import basic_module_init
from .norm import ReparamBatchNorm2d, apply_batch_max_iteration_, apply_frozen_bn_  # noqa
from .replication_pad2d import ReplicationPad2dNaive


def _fit_to_size(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    dy = (x.shape[-2] - h) // 2
    dx = (x.shape[-1] - w) // 2
    return x[..., dy : dy + h, dx : dx + w]


class ReparamBranch(nn.Module):
    def __init__(self, groups: int = 1):
        super().__init__()
        self.groups = groups

    def fuse(self):
        raise NotImplementedError

    def get_kernel_size(self):
        raise NotImplementedError


class Shortcut(ReparamBranch):
    conv: None | nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int | None = None, groups: int = 1):
        super().__init__(groups=groups)
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        assert in_channels % groups == 0 and out_channels % groups == 0

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        else:
            return x

    def fuse(self):
        if self.conv is not None:
            w = self.conv.weight.double()
            b = torch.zeros(self.out_channels, dtype=torch.double, device=w.device)
        else:
            # identity
            assert self.in_channels == self.out_channels
            w = torch.zeros((self.out_channels, self.in_channels // self.groups, 1, 1), dtype=torch.double)
            for i in range(self.out_channels):
                w[i, i % (self.in_channels // self.groups), 0, 0] = 1.0
            b = torch.zeros(self.out_channels, dtype=torch.double)
        return w, b

    def get_kernel_size(self):
        return (1, 1)


class Conv2dBranch(ReparamBranch):
    dropout: nn.Dropout | nn.Identity

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        groups: int = 1,
        use_bn=False,
        bn_layer=nn.BatchNorm2d,
        dropout_p=0.0,
    ):
        super().__init__(groups=groups)
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=not use_bn
        )
        if use_bn:
            self.bn = bn_layer(out_channels)
        else:
            self.bn = nn.Identity()
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def forward(self, x):
        return self.dropout(self.bn(self.conv(x)))

    def fuse(self):
        if not isinstance(self.bn, nn.Identity):
            conv = fuse_conv_bn_eval(self.conv, self.bn)
        else:
            conv = self.conv

        w = conv.weight.double()
        b = (
            conv.bias.double()
            if conv.bias is not None
            else torch.zeros(self.out_channels, dtype=torch.double, device=w.device)
        )
        return w, b

    def get_kernel_size(self):
        return self.kernel_size


class Parallel(ReparamBranch):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        structures,
        groups: int = 1,
        use_bn=False,
        bn_layer=nn.BatchNorm2d,
        dropout_p=0.0,
    ):
        super().__init__(groups=groups)
        if isinstance(use_bn, bool):
            use_bn = [use_bn] * len(structures)
        if isinstance(dropout_p, (int, float)):
            dropout_p = [dropout_p] * len(structures)
        self.use_bn = use_bn
        self.bn_layer = bn_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branches = nn.ModuleList(
            [
                self._build(in_channels, out_channels, s, groups=groups, use_bn=self.use_bn[i], dropout_p=dropout_p[i])
                for i, s in enumerate(structures)
            ]
        )

    def _build(self, in_c, out_c, s, groups, use_bn, dropout_p):
        if isinstance(s, (int, tuple)):
            return Conv2dBranch(
                in_c, out_c, s, groups=groups, use_bn=use_bn, bn_layer=self.bn_layer, dropout_p=dropout_p
            )
        if isinstance(s, ReparamBranch):
            assert s.groups == groups, f"nested structure.groups ({s.groups}) must match parent groups ({groups})"
            return s
        raise ValueError(f"Unsupported structure: {s}")

    def forward(self, x):
        outputs = [b(x) for b in self.branches]
        min_h = min(o.shape[-2] for o in outputs)
        min_w = min(o.shape[-1] for o in outputs)

        res = 0
        for i, o in enumerate(outputs):
            o = _fit_to_size(o, min_h, min_w)
            res = res + o
        return res

    def fuse(self):
        fused = [b.fuse() for b in self.branches]
        max_h, max_w = self.get_kernel_size()

        # Find device
        device = torch.device("cpu")
        for w, b in fused:
            if w.device.type != "cpu":
                device = w.device
                break

        res_w = torch.zeros(
            (self.out_channels, self.in_channels // self.groups, max_h, max_w), dtype=torch.double, device=device
        )
        res_b = torch.zeros(self.out_channels, dtype=torch.double, device=device)

        for i, (w, b) in enumerate(fused):
            ph = (max_h - w.shape[-2]) // 2
            pw = (max_w - w.shape[-1]) // 2
            res_w += F.pad(w.to(device), [pw, pw, ph, ph])
            res_b += b.to(device)
        return res_w, res_b

    def get_kernel_size(self):
        sizes = [b.get_kernel_size() for b in self.branches]
        return max(s[0] for s in sizes), max(s[1] for s in sizes)


class Series(ReparamBranch):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        structures,
        groups: int = 1,
        middle_factor: int | float = 1,
        use_bn=False,
        bn_layer=nn.BatchNorm2d,
        dropout_p=0.0,
    ):
        super().__init__(groups=groups)
        if isinstance(use_bn, bool):
            use_bn = [use_bn] * len(structures)
        if isinstance(dropout_p, (int, float)):
            dropout_p = [dropout_p] * len(structures)
        self.use_bn = use_bn
        self.bn_layer = bn_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert in_channels % groups == 0 and out_channels % groups == 0

        # Scaled relative to out_channels
        self.middle_channels = int(out_channels * middle_factor)
        if groups > 1:
            self.middle_channels = max(self.middle_channels - self.middle_channels % groups, groups)
        else:
            self.middle_channels = max(self.middle_channels - self.middle_channels % 8, 8)

        branches = []
        num_branches = len(structures)
        for i, s in enumerate(structures):
            branch_in = in_channels if i == 0 else self.middle_channels
            branch_out = out_channels if i == num_branches - 1 else self.middle_channels
            branches.append(
                self._build(branch_in, branch_out, s, groups=groups, use_bn=self.use_bn[i], dropout_p=dropout_p[i])
            )
        self.branches = nn.ModuleList(branches)

    def _build(self, in_c, out_c, s, groups, use_bn, dropout_p):
        if isinstance(s, (int, tuple)):
            return Conv2dBranch(
                in_c, out_c, s, groups=groups, use_bn=use_bn, bn_layer=self.bn_layer, dropout_p=dropout_p
            )
        if isinstance(s, ReparamBranch):
            assert s.groups == groups, f"nested structure.groups ({s.groups}) must match parent groups ({groups})"
            return s
        raise ValueError(f"Unsupported structure: {s}")

    def forward(self, x):
        for b in self.branches:
            x = b(x)
        return x

    def fuse(self):
        curr_w, curr_b = self.branches[0].fuse()
        for i in range(1, len(self.branches)):
            next_w, next_b = self.branches[i].fuse()
            device = next_w.device if next_w.device.type != "cpu" else curr_w.device
            curr_w = curr_w.to(device)
            curr_b = curr_b.to(device)
            next_w = next_w.to(device)
            next_b = next_b.to(device)

            # Weight fusion
            fused_w = F.conv2d(
                curr_w.transpose(0, 1),
                next_w.flip(-1, -2),
                groups=self.groups,
                padding=(next_w.shape[-2] - 1, next_w.shape[-1] - 1),
            )
            curr_w = fused_w.transpose(0, 1)

            # Bias fusion
            next_w_sum = next_w.sum(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
            curr_b = F.conv2d(curr_b.reshape(1, -1, 1, 1), next_w_sum, groups=self.groups).reshape(-1) + next_b

        return curr_w, curr_b

    def get_kernel_size(self):
        sizes = [b.get_kernel_size() for b in self.branches]
        h = sum(s[0] for s in sizes) - (len(sizes) - 1)
        w = sum(s[1] for s in sizes) - (len(sizes) - 1)
        return h, w


class ReparamConv2d(nn.Module):
    padding: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        structure,
        padding: bool = False,
        groups: int = 1,
        use_bn=False,
        bn_layer=nn.BatchNorm2d,
        dropout_p=0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if isinstance(structure, (list, tuple)):
            self.structure = Parallel(
                in_channels,
                out_channels,
                structure,
                groups=groups,
                use_bn=use_bn,
                bn_layer=bn_layer,
                dropout_p=dropout_p,
            )
        else:
            if hasattr(structure, "groups"):
                assert structure.groups == groups, f"structure.groups ({structure.groups}) must match groups ({groups})"
            self.structure = structure

        kernel_size = self.structure.get_kernel_size()
        if padding:
            pad_h = kernel_size[0] - 1
            pad_w = kernel_size[1] - 1
            self.padding = ReplicationPad2dNaive((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.padding = nn.Identity()

        self.conv_eval = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=0)
        basic_module_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)

        if not self.training:
            return self.conv_eval(x)

        return self.structure(x)

    def train(self, mode: bool = True):
        super().train(mode)

        if not mode:
            return self.fuse()

        return self

    def fuse(self) -> nn.Conv2d:
        assert self.conv_eval.bias is not None
        w, b = self.structure.fuse()
        dtype = self.conv_eval.weight.dtype
        self.conv_eval.weight.data.copy_(w.to(dtype))
        self.conv_eval.bias.data.copy_(b.to(dtype))
        return self.conv_eval

    def purge(self) -> nn.Module:
        delattr(self, "structure")
        return self

    def get_kernel_size(self):
        return self.structure.get_kernel_size()


def apply_fuse_(model: nn.Module):
    for name, child in model.named_children():
        if isinstance(child, ReparamConv2d):
            setattr(model, name, child.fuse())
        else:
            apply_fuse_(child)


def _test(dtype=torch.float64):
    print("******", dtype)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_c, out_c = 64, 64

    # Structure definition with middle_factor relative to out_channels
    # bn_layer = nn.BatchNorm2d
    bn_layer = ReparamBatchNorm2d
    structure = [
        1,
        3,
        5,
        7,
        (1, 7),
        (7, 1),
        Shortcut(in_c),
        Series(in_c, out_c, [1, 3, 1], middle_factor=2.0, use_bn=(False, True, False), bn_layer=bn_layer),
    ]
    model = ReparamConv2d(in_c, out_c, structure, padding=True, use_bn=True, bn_layer=bn_layer).to(device, dtype=dtype)
    apply_batch_max_iteration_(model, 100)

    print("*** train", model)
    x = torch.rand((4, in_c, 64, 64)).to(device, dtype=dtype)

    model.train()
    for i in range(100):
        model(x)

    apply_frozen_bn_(model)

    z1 = model(x)
    model.eval()
    model.purge()
    print("*** eval", model)
    z2 = model(x)

    diff = (z1 - z2).abs().max().item()
    print(f"Max difference: {diff}")
    if diff < 1e-5:
        print("Test Passed")
    else:
        print("Test Failed")

    # test Shortcut with channel mismatch
    in_c2, out_c2 = 16, 32
    structure2 = [
        Shortcut(in_c2, out_c2),
        Conv2dBranch(in_c2, out_c2, 3),
    ]
    model = ReparamConv2d(in_c2, out_c2, structure2, padding=True).to(device, dtype=dtype)
    x = torch.rand((4, in_c2, 64, 64)).to(device, dtype=dtype)
    model.train()
    z1 = model(x)
    model.eval()
    z2 = model(x)
    diff = (z1 - z2).abs().max().item()
    print(f"Shortcut(mismatch) Max difference: {diff}")
    if diff < 1e-5:
        print("Shortcut(mismatch) Test Passed")
    else:
        print("Shortcut(mismatch) Test Failed")

    # test Depthwise
    in_c3, out_c3, groups3 = 32, 32, 32
    structure3 = [
        3,
        5,
        Shortcut(in_c3, out_c3, groups=groups3),
        Series(in_c3, out_c3, [1, 3, 1], groups=groups3),
    ]
    model = ReparamConv2d(in_c3, out_c3, structure3, groups=groups3, padding=True).to(device, dtype=dtype)
    x = torch.rand((4, in_c3, 64, 64)).to(device, dtype=dtype)
    model.train()
    z1 = model(x)
    model.eval()
    z2 = model(x)
    diff = (z1 - z2).abs().max().item()
    print(f"Depthwise Max difference: {diff}")
    if diff < 1e-5:
        print("Depthwise Test Passed")
    else:
        print("Depthwise Test Failed")

    # test Grouped
    in_c4, out_c4, groups4 = 64, 128, 4
    structure4 = [
        3,
        Shortcut(in_c4, out_c4, groups=groups4),
        Series(in_c4, out_c4, [3, 3], groups=groups4),
    ]
    model = ReparamConv2d(in_c4, out_c4, structure4, groups=groups4, padding=True).to(device, dtype=dtype)
    x = torch.rand((4, in_c4, 64, 64)).to(device, dtype=dtype)
    model.train()
    z1 = model(x)
    model.eval()
    z2 = model(x)
    diff = (z1 - z2).abs().max().item()
    print(f"Grouped Max difference: {diff}")
    if diff < 1e-5:
        print("Grouped Test Passed")
    else:
        print("Grouped Test Failed")


if __name__ == "__main__":
    _test(torch.float64)
    # _test(torch.float32)
    # _test(torch.float16)
    # _test(torch.bfloat16)
