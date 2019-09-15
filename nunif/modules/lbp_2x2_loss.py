import torch
import torch.nn as nn
from . weighted_huber_loss import WeightedHuberLoss


LBP =[
# identify
[[  1,  0],
 [  0,  0]],
# diff    
[[ -1,  1],
 [  1, -1]],
[[ -1,  1],
 [ -1,  1]],
[[ -1, -1],
 [  1,  1]],
[[ -1,  0],
 [  0,  1]],
[[  0, -1],
 [  1,  0]],
[[ -1,  1],
 [  0,  0]],
[[ -1,  0],
 [  1,  0]]
]


class LBP2x2Loss(nn.Module):
    def __init__(self, ch, gamma=0.1):
        assert(ch == 1 or ch == 3)
        super(LBP2x2Loss, self).__init__()
        lbp_kernel = torch.FloatTensor(LBP).view(len(LBP), 1, 2, 2)
        for i in range(len(LBP)):
            wsum = lbp_kernel[i].sum().item()
            if abs(wsum) > 0:
                lbp_kernel[i].div_(wsum)
        if ch == 1:
            self.lbp = nn.Conv2d(ch, len(LBP), kernel_size=2, stride=1, padding=0, bias=False)
            self.lbp.weight.data.copy_(lbp_kernel)
            print(self.lbp.weight.data)
        elif ch == 3:
            self.lbp = nn.Conv2d(ch, len(LBP) * 3, kernel_size=2, stride=1, padding=0, bias=False)
            self.lbp.weight.data.fill_(0)
            for i in range(len(LBP)):
                r = i
                g = i + len(LBP)
                b = i + len(LBP) * 2
                self.lbp.weight.data[r][0].copy_(lbp_kernel[i][0])
                self.lbp.weight.data[g][1].copy_(lbp_kernel[i][0])
                self.lbp.weight.data[b][2].copy_(lbp_kernel[i][0])
        if ch == 1:
            channel_weight = torch.FloatTensor([1])
        else:
            channel_weight = torch.FloatTensor([0.299, 0.587, 0.114])
        self.loss_module = WeightedHuberLoss(channel_weight, gamma=gamma, reduction='spatial_mean')

    def _debug_forward(self, x):
        return self.lbp(x)

    def forward(input, target):
        return self.loss_module(self.lbp(input), self.lbp(target))

if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import functional as TF
    from torchvision.utils import save_image

    im = Image.open("tmp/color.jpg").convert("RGB")
    x = TF.to_tensor(TF.to_grayscale(im)).unsqueeze(0).float()
    loss = LBP2x2Loss(1, nn.L1Loss())
    z = loss._debug_forward(x).squeeze(0)
    z = z.view(z.shape[0], 1, z.shape[1], z.shape[2])
    save_image(z, "tmp/lbp.png", nrow=8)

    im = Image.open("tmp/color.jpg").convert("RGB")
    x = TF.to_tensor(im).unsqueeze(0).float()
    loss = LBP2x2Loss(3, nn.L1Loss())
    z = loss._debug_forward(x).squeeze(0)
    z = z.view(z.shape[0], 1, z.shape[1], z.shape[2])
    save_image(z, "tmp/lbp_rgb.png", nrow=8)

