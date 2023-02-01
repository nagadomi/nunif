# SwinTransformer with torchvision.models
# 93.7% accuracy on CIFAR10 using only CIFAR10 training data,
# python3 -m playground.vit.train_cifar10_swin --data-dir ./tmp/vit --model-dir ./tmp/vit
from torchvision.models import SwinTransformer
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torch
from torch.nn import functional as F
from nunif.models import SoftmaxBaseModel
from nunif.training.env import SoftmaxEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser
import nunif.transforms as NT


def normalize(x):
    return (x - 0.5) * 2


class Normalize():
    def __call__(self, x):
        return normalize(x)


IMG_SIZE = 64


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train):
        super().__init__()
        self.train = train
        if train:
            t1 = T.Compose([
                T.RandomCrop((24, 24)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t2 = T.Compose([
                T.RandomCrop((26, 26)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t3 = T.Compose([
                T.RandomCrop((28, 28)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t4 = T.Compose([
                T.RandomCrop((30, 30)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            transform = T.Compose([
                T.RandomChoice([NT.Identity(),
                                T.Compose([
                                    NT.ReflectionResize((38, 38)),
                                    T.RandomPerspective(distortion_scale=0.15, p=1),
                                    T.CenterCrop((32, 32))])]),
                T.RandomChoice([t1, t2, t3, t4]),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                Normalize(),
            ])
        else:
            # TTA at __getitem__
            transform = None
        self.cifar10 = CIFAR10(root, train, transform=transform, download=True)

    def __len__(self):
        return len(self.cifar10)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def show(self, im):
        from nunif.utils.pil_io import to_cv2
        import cv2
        cv2.imshow("debug", to_cv2(im))
        cv2.waitKey(0)

    def __getitem__(self, i):
        x, y = self.cifar10[i]
        if self.train:
            return x, y
        else:
            tta = []
            tta += [TF.resize(crop, (IMG_SIZE, IMG_SIZE)) for crop in TF.five_crop(x, (24, 24))]
            tta += [TF.resize(crop, (IMG_SIZE, IMG_SIZE)) for crop in TF.five_crop(x, (28, 28))]
            tta += [TF.resize(x, (IMG_SIZE, IMG_SIZE))]
            tta = tta + [TF.hflip(crop) for crop in tta]
            x = torch.stack([normalize(TF.to_tensor(crop)) for crop in tta], dim=0)
            return x, y


CIFAR10_CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VIT(SoftmaxBaseModel):
    name = "swintransformer"

    def __init__(self):
        super().__init__(locals(), CIFAR10_CLASS_NAMES)
        self.swin_transformer = SwinTransformer(
            num_classes=len(CIFAR10_CLASS_NAMES),
            patch_size=[4, 4],  # 64x64 -> 16x16
            embed_dim=64,
            # 16x16, 8x8, 4x4
            depths=[2, 6, 2],
            num_heads=[4, 8, 16],
            window_size=[4, 4],
            stochastic_depth_prob=0.2,
        )
        #  print(self.swin_transformer)

    def forward(self, x):
        z = self.swin_transformer(x)
        if self.training:
            return F.log_softmax(z, dim=1)
        else:
            return F.softmax(z, dim=1)


class CIFAR10Trainer(Trainer):
    def create_model(self):
        model = VIT().to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = CIFAR10Dataset(self.args.data_dir, train=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader
        else:
            dataset = CIFAR10Dataset(self.args.data_dir, train=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size // 4,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        return SoftmaxEnv(self.model, eval_tta=True)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.set_defaults(
        batch_size=128,
        num_workers=4,
        scheduler="cosine",
        optimizer="adamw",
        learning_rate=0.00015,
        learning_rate_decay=0.99,
        warmup_epoch=2,
        warmup_learning_rate=1e-7,
        max_epoch=200,
        disable_amp=False
    )
    args = parser.parse_args()
    trainer = CIFAR10Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
