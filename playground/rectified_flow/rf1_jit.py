# Rectified Flow JiT (x-prediction)
import os
import sys
from os import path

# compile
os.environ["NUNIF_TRAIN"] = "1"


import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid

from nunif.modules.charbonnier_loss import CharbonnierLoss
from nunif.training.env import I2IEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser

from .celeba_dataset import IMAGE_SIZE, CelebADataset
from .dit import DiT
from .dit_self_cond import DiTSelfCond
from .dit_without_cond import DiTWithoutCond


def predict_velocity(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # From x-prediction to v
    z = model(x_t, t)
    denom = (1.0 - t.view(-1, 1, 1, 1)).clamp(min=0.05)
    v = (z - x_t) / denom
    return v, z


@torch.inference_mode()
def sample(model: nn.Module, x: torch.Tensor, steps: int = 25) -> torch.Tensor:
    B = x.shape[0]
    # linear timestep (euler simple)
    t_steps = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    for i in range(steps):
        # step size
        t = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t
        # Run step
        v, _ = predict_velocity(model, x, t.expand(B))
        x = x + v * dt

    # denormalize: -1 .. 1 to 0 .. 1
    x = ((x + 1.0) * 0.5).clamp(0, 1)
    return x


class RFEnv(I2IEnv):
    def __init__(self, model, criterion):
        super().__init__(model, criterion=criterion, eval_criterion=criterion)
        self.eval_rng = torch.Generator(device=self.device)

    def train_step(self, data):
        x, *_ = data
        # x: -1 .. 1
        x = self.to_device(x)
        with self.autocast():
            # Generate a 0-1 timestep
            t = torch.sigmoid(torch.randn((x.shape[0],), device=x.device) * 1)
            w = t.view(-1, 1, 1, 1)
            # Generate a noise
            noise = torch.randn_like(x)
            # Blend the current state
            x_t = (1 - w) * noise + w * x
            denom = (1.0 - w).clamp(min=0.05)
            # From x-prediction to v:
            #   v = (x - ((1 - w) * noise + w * x)) / (1 - w) = x - noise
            target_velocity = (x - x_t) / denom
            z = self.model(x_t, t)
            v = (z - x_t) / denom
            loss = self.criterion(v, target_velocity)

        if not torch.any(torch.isnan(loss)):
            self.sum_loss += loss.detach()
            self.sum_step += 1
        return loss

    def eval_begin(self):
        self.eval_rng.manual_seed(42)
        super().eval_begin()

    def eval_step(self, data):
        x, *_ = data
        x = self.to_device(x)
        model = self.get_eval_model()
        with self.autocast():
            t = torch.sigmoid(torch.randn((x.shape[0],), device=x.device) * 1)
            w = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(x, generator=self.eval_rng)
            x_t = (1 - w) * noise + w * x
            denom = (1.0 - w).clamp(min=0.05)
            target_velocity = (x - x_t) / denom
            z = model(x_t, t)
            v = (z - x_t) / denom
            loss = self.eval_criterion(v, target_velocity)
        self.sum_loss += loss.detach()
        self.sum_step += 1

    def eval_end(self, file=sys.stdout):
        model = self.get_eval_model()
        self.eval_rng.manual_seed(42)
        with self.autocast():
            outputs = sample(
                model,
                torch.randn((16, 3, IMAGE_SIZE, IMAGE_SIZE), generator=self.eval_rng, device=self.device),
            )
        output_file = path.join(self.trainer.args.model_dir, f"sample_{self.trainer.epoch}.png")
        TF.to_pil_image(make_grid(outputs, nrow=8)).save(output_file)

        return super().eval_end(file=file)


class RFTrainer(Trainer):
    def create_dataloader(self, type):
        assert type in {"train", "eval"}
        if type == "train":
            dataset = CelebADataset(self.args.data_dir, split="train")

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True,
            )
            return loader
        else:
            dataset = CelebADataset(self.args.data_dir, split="test")
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True,
            )
            return loader

    def create_model(self):
        if self.args.without_cond:
            model = DiTWithoutCond().to(self.device)
        elif self.args.self_cond:
            model = DiTSelfCond().to(self.device)
        else:
            model = DiT().to(self.device)
        return model

    def create_env(self):
        if self.args.loss == "mse":
            criterion = torch.nn.MSELoss()
        elif self.args.loss == "char":
            criterion = CharbonnierLoss()

        return RFEnv(self.model, criterion=criterion)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=40000)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "char"])
    parser.add_argument("--without-cond", action="store_true", help="without timestep conditioning")
    parser.add_argument(
        "--self-cond", action="store_true", help="use self conditioning instead of timestep conditioning"
    )
    parser.set_defaults(
        batch_size=32,
        num_workers=8,
        max_epoch=400,
        learning_rate=0.001,
        optimizer="adamw_schedulefree",
        warmup_epoch=1,
        eval_step=4,
    )
    args = parser.parse_args()
    trainer = RFTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
