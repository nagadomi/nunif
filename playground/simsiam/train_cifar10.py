# A rough implementation of SimSiam https://arxiv.org/abs/2011.10566
# for CIFAR10
# Current implementation only achieves CIFAR10 80% accuracy with linear classifier
# python -m playground.simsiam.train_cifar10 --data-dir ./data/cifar10 --model-dir ./models/simsima
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torch
from torch import nn
from torch.nn import functional as F
from nunif.models import Model, get_model_device
from nunif.modules.res_block import ResBlockBNLReLU
from nunif.modules.init import basic_module_init
from nunif.modules.gaussian_filter import GaussianFilter2d
from nunif.training.env import BaseEnv
from nunif.training.confusion_matrix import SoftmaxConfusionMatrix
from nunif.training.trainer import Trainer, create_trainer_default_parser


IMG_SIZE = 32


def normalize(x):
    return (x - 0.5) * 2


class Normalize():
    def __call__(self, x):
        return normalize(x)


class Identity():
    def __call__(self, x):
        return x


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train):
        super().__init__()
        self.train = train
        self.cifar10 = CIFAR10(root, train, transform=None, download=True)
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.)),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.RandomGrayscale(p=0.1),
                # T.RandomChoice([Identity(), T.GaussianBlur(3), T.GaussianBlur(5)], p=(0.5, 0.25, 0.25)),
                T.ToTensor(),
                Normalize(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                Normalize(),
            ])

    def __len__(self):
        return len(self.cifar10)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def __getitem__(self, i):
        x, y = self.cifar10[i]
        if self.train:
            x1, x2 = self.transform(x), self.transform(TF.hflip(x))
            return x1, x2, y
        else:
            x = self.transform(x)
            return x, y


class SimSiam(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        # NOTE: Without BN, this collapsed quickly
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlockBNLReLU(32, 32, bias=False),
            ResBlockBNLReLU(32, 64, stride=2, bias=False),    # 16x16
            ResBlockBNLReLU(64, 64, bias=False),
            ResBlockBNLReLU(64, 128, stride=2, bias=False),   # 8x8
            ResBlockBNLReLU(128, 128, bias=False),
            ResBlockBNLReLU(128, 256, stride=2, bias=False),  # 4x4
            ResBlockBNLReLU(256, 256, bias=False),
            nn.Conv2d(256, dim, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(dim, dim, bias=False),
        )
        basic_module_init(self)

    def forward(self, x1, x2):
        B = x1.shape[0]
        z1 = self.encoder(x1).view(B, -1)
        z2 = self.encoder(x2).view(B, -1)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class LinearClassifier(nn.Module):
    def __init__(self, dim=256, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        basic_module_init(self)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        z = self.linear(x)
        if self.training:
            return F.log_softmax(z, dim=1)
        else:
            return F.softmax(z, dim=1)


class SimSiamEnvModel(Model):
    def __init__(self):
        super().__init__({})
        self.simsiam = SimSiam(256)
        self.classifier = LinearClassifier(dim=256)

    def forward(self, x1, x2):
        p1, p2, z1, z2 = self.simsiam(x1, x2)
        z3 = self.classifier(z1.detach())

        return p1, p2, z1, z2, z3


class SimSiamEnv(BaseEnv):
    def __init__(self, model, eval_tta=False):
        super().__init__()
        self.eval_tta = eval_tta
        self.model = model
        self.device = get_model_device(self.model)
        self.similarity_criterion = nn.CosineSimilarity(dim=1).to(self.device)
        self.classifier_criterion = nn.NLLLoss().to(self.device)
        self.confusion_matrix = SoftmaxConfusionMatrix([str(i) for i in range(10)])
        self.t = 0

    def clear_loss(self):
        self.sum_sim_loss = 0
        self.sum_class_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        x1, x2, y, *_ = data
        x1, x2, y = self.to_device(x1), self.to_device(x2), self.to_device(y)
        with self.autocast():
            p1, p2, z1, z2, z3 = self.model(x1, x2)
            sim_loss = (self.similarity_criterion(p1, z2).mean() + self.similarity_criterion(p2, z1).mean()) * -0.5
            class_loss = self.classifier_criterion(z3, y)

        self.sum_sim_loss += sim_loss.item()
        self.sum_class_loss += class_loss.item()
        self.sum_step += 1
        return sim_loss + class_loss

    def train_end(self):
        mean_sim_loss = self.sum_sim_loss / self.sum_step
        mean_class_loss = self.sum_class_loss / self.sum_step
        print(f"loss: sim={mean_sim_loss}, class={mean_class_loss}")
        return mean_sim_loss

    def eval_begin(self):
        self.model.eval()
        self.confusion_matrix.clear()

    def eval_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        B = x.shape[0]
        with self.autocast():
            x = self.model.simsiam.encoder(x).view(B, -1)
            z = self.model.classifier(x)
            labels = torch.argmax(z, dim=1).cpu()
            self.confusion_matrix.update(labels, y)

    def eval_end(self):
        self.confusion_matrix.print()
        return 1 - self.confusion_matrix.average_row_correct()


class CIFAR10Trainer(Trainer):
    def create_model(self):
        model = SimSiamEnvModel().to(self.device)
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
        return SimSiamEnv(self.model, eval_tta=True)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=60000)
    parser.set_defaults(
        batch_size=128,
        num_workers=8,
        optimizer="sgd",
        learning_rate=0.05,
        learning_rate_decay=0.985,
        scheduler="cosine",
        learning_rate_cycles=4,
        max_epoch=400,
        skip_eval=4,
        disable_amp=False,
    )
    args = parser.parse_args()
    trainer = CIFAR10Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
