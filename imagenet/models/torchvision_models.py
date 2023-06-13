import torch
from torchvision.models.vgg import vgg11_bn, VGG11_BN_Weights
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from nunif.models import SoftmaxBaseModel, register_model
import torch.nn.functional as F
from ..class_names import CLASS_NAMES


@register_model
class VGG11BN(SoftmaxBaseModel):
    name = "torchvision.vgg11_bn"

    def __init__(self, pretrained=False):
        super().__init__(locals(), class_names=CLASS_NAMES)
        if pretrained:
            self.net = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        else:
            self.net = vgg11_bn()

    def forward(self, x):
        z = self.net(x)
        if self.training:
            return F.log_softmax(z, dim=1)
        else:
            return F.softmax(z, dim=1)



@register_model
class SwinT(SoftmaxBaseModel):
    name = "torchvision.swin_t"

    def __init__(self, pretrained=False):
        super().__init__(locals(), class_names=CLASS_NAMES)
        if pretrained:
            self.net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            self.net = swin_t()

    def forward(self, x):
        z = self.net(x)
        if self.training:
            return F.log_softmax(z, dim=1)
        else:
            return F.softmax(z, dim=1)


if __name__ == "__main__":
    print(VGG11BN())
    print(SwinT())
