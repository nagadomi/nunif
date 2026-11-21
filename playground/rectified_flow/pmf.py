# Pixel Mean Flow (x-prediction)
import os

os.environ["NUNIF_TRAIN"] = "1"
import argparse
import sys
from os import path
from typing import Any, TextIO

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid

from nunif.modules.attention import use_math_attention
from nunif.modules.charbonnier_loss import CharbonnierLoss
from nunif.modules.lpips import LPIPSLoss
from nunif.training.env import I2IEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser

from .celeba_dataset import IMAGE_SIZE, CelebADataset
from .dit_pmf import DiTPMF


def predict_u(model: DiTPMF, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    x_u = model.forward_u(x_t, t, h)
    denom = t.view(-1, 1, 1, 1).clamp(min=0.05)
    return (x_t - x_u) / denom


def _predict_v_tangent(model: DiTPMF, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x_v = model.forward_v(x_t, t)
    denom = t.view(-1, 1, 1, 1).clamp(min=0.05)
    return (x_t - x_v) / denom


@torch.inference_mode()
def sample(model: DiTPMF, x: torch.Tensor, steps: int = 1, shift: float = 1.0) -> torch.Tensor:
    assert steps >= 1
    assert shift > 0.0

    B = x.shape[0]
    # pMF integrates from t = 1 (noise) to t = 0 (data)
    s = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    s = s / (s + shift * (1.0 - s))
    t_steps = 1.0 - s

    for i in range(steps):
        t = t_steps[i]
        r = t_steps[i + 1]
        h = t - r
        u = predict_u(model, x, t.expand(B), h.expand(B))
        x = x - h * u

    # denormalize: -1 .. 1 to 0 .. 1
    x = ((x + 1.0) * 0.5).clamp(0, 1)
    return x


def adp_wt_fn(loss: torch.Tensor, norm_p: float, norm_eps: float) -> torch.Tensor:
    if norm_p == 0.0:
        return loss
    adp_wt = (loss + norm_eps) ** norm_p
    return loss / adp_wt.detach()


class PMFEnv(I2IEnv):
    def __init__(
        self,
        model: DiTPMF,
        loss_type: str = "mse",
        p_mean: float = 0.8,
        p_std: float = 0.8,
        data_proportion: float = 0.5,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        lpips_lambda: float = 0.4,
        perceptual_max_t: float = 0.8,
        sample_steps: int = 1,
    ) -> None:
        if loss_type == "char":
            criterion: nn.Module = CharbonnierLoss(reduction="none")
        else:
            criterion = nn.MSELoss(reduction="none")
        super().__init__(model, criterion=criterion, eval_criterion=criterion)

        self.p_mean = p_mean
        self.p_std = p_std
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.lpips_lambda = lpips_lambda
        self.perceptual_max_t = perceptual_max_t
        self.sample_steps = sample_steps

        self.lpips: LPIPSLoss | None
        if self.lpips_lambda > 0.0:
            self.lpips = LPIPSLoss(normalize=False, reduction="none").to(self.device)
        else:
            self.lpips = None

        self.eval_rng = torch.Generator(device=self.device)
        self.sum_loss_u: float | torch.Tensor = 0.0
        self.sum_loss_v: float | torch.Tensor = 0.0
        self.sum_loss_lpips: float | torch.Tensor = 0.0

    def clear_loss(self) -> None:
        super().clear_loss()
        self.sum_loss_u = 0.0
        self.sum_loss_v = 0.0
        self.sum_loss_lpips = 0.0

    def train_begin(self) -> None:
        super().train_begin()
        if self.lpips is not None:
            self.lpips.eval()
            self.lpips.requires_grad_(False)

    def _sample_tr(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.sigmoid(
            torch.randn((batch_size,), device=self.device, generator=generator) * self.p_std + self.p_mean
        )
        r = torch.sigmoid(
            torch.randn((batch_size,), device=self.device, generator=generator) * self.p_std + self.p_mean
        )
        data_size = int(batch_size * self.data_proportion)
        if data_size > 0:
            r[:data_size] = t[:data_size]
        t_out = torch.maximum(t, r)
        r_out = torch.minimum(t, r)
        return t_out, r_out

    def _compute_step_losses(
        self,
        model: DiTPMF,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        t, r = self._sample_tr(B, generator=generator)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        w = t.view(-1, 1, 1, 1)
        z_t = (1.0 - w) * x + w * noise
        denom = w.clamp(min=0.05)
        v_target = (z_t - x) / denom

        with torch.no_grad():
            v_c = _predict_v_tangent(model, z_t, t).detach()

        def u_fn(z_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_u, x_v = model(z_in, t_in, t_in - r_in)
            d = t_in.view(-1, 1, 1, 1).clamp(min=0.05)
            return (z_in - x_u) / d, (z_in - x_v) / d

        ones = torch.ones_like(t)
        zeros = torch.zeros_like(r)
        with use_math_attention():
            jvp_res: Any = torch.func.jvp(u_fn, (z_t, t, r), (v_c, ones, zeros), has_aux=True)
            u, du_dt, v = jvp_res

        V = u + (t - r).view(-1, 1, 1, 1) * du_dt.detach()

        err_u = self.criterion(V.float(), v_target.float())
        err_v = self.criterion(v.float(), v_target.float())
        loss_u = adp_wt_fn(err_u.sum(dim=(1, 2, 3)), self.norm_p, self.norm_eps)
        loss_v = adp_wt_fn(err_v.sum(dim=(1, 2, 3)), self.norm_p, self.norm_eps)
        total = loss_u + loss_v

        if self.lpips is not None:
            pred_x = z_t - w * u
            lpips_per_sample = self.lpips(pred_x, x).float()
            lpips_per_sample = torch.where(t < self.perceptual_max_t, lpips_per_sample, 0.0)
            total = total + adp_wt_fn(lpips_per_sample, self.norm_p, self.norm_eps) * self.lpips_lambda
            metric_lpips = lpips_per_sample.mean().detach()
        else:
            metric_lpips = torch.zeros((), device=self.device)

        return total.mean(), err_u.mean().detach(), err_v.mean().detach(), metric_lpips

    def _accumulate_metrics(self, metric_u: torch.Tensor, metric_v: torch.Tensor, metric_lpips: torch.Tensor) -> None:
        self.sum_loss += metric_u + metric_v + metric_lpips * self.lpips_lambda
        self.sum_loss_u += metric_u
        self.sum_loss_v += metric_v
        self.sum_loss_lpips += metric_lpips
        self.sum_step += 1

    def train_step(self, data: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, *_ = data
        x = self.to_device(x)
        with self.autocast():
            loss, metric_u, metric_v, metric_lpips = self._compute_step_losses(self.model, x)

        if not torch.any(torch.isnan(loss)):
            self._accumulate_metrics(metric_u, metric_v, metric_lpips)
        return loss

    def train_end(self) -> float:
        mean_loss = self.to_scalar(self.sum_loss / self.sum_step)
        mean_u = self.to_scalar(self.sum_loss_u / self.sum_step)
        mean_v = self.to_scalar(self.sum_loss_v / self.sum_step)
        if self.lpips is not None:
            mean_lpips = self.to_scalar(self.sum_loss_lpips / self.sum_step)
            print(f"loss: {mean_loss:.6f} (u: {mean_u:.6f}, v: {mean_v:.6f}, lpips: {mean_lpips:.6f})")
        else:
            print(f"loss: {mean_loss:.6f} (u: {mean_u:.6f}, v: {mean_v:.6f})")
        return mean_loss

    def eval_begin(self) -> None:
        self.eval_rng.manual_seed(42)
        super().eval_begin()

    def eval_step(self, data: tuple[torch.Tensor, ...]) -> None:
        assert not self.trainer.args.ema_model, "EMA is not supported in PMFEnv"
        x, *_ = data
        x = self.to_device(x)
        model = self.get_eval_model()
        assert isinstance(model, DiTPMF)
        with self.autocast():
            _, metric_u, metric_v, metric_lpips = self._compute_step_losses(model, x, generator=self.eval_rng)
        self._accumulate_metrics(metric_u, metric_v, metric_lpips)

    def print_eval_result(self, loss: float | torch.Tensor, file: TextIO = sys.stdout) -> None:
        mean_loss = self.to_scalar(loss)
        mean_u = self.to_scalar(self.sum_loss_u / self.sum_step)
        mean_v = self.to_scalar(self.sum_loss_v / self.sum_step)
        if self.lpips is not None:
            mean_lpips = self.to_scalar(self.sum_loss_lpips / self.sum_step)
            print(f"loss: {mean_loss:.6f} (u: {mean_u:.6f}, v: {mean_v:.6f}, lpips: {mean_lpips:.6f})", file=file)
        else:
            print(f"loss: {mean_loss:.6f} (u: {mean_u:.6f}, v: {mean_v:.6f})", file=file)

    def eval_end(self, file: TextIO = sys.stdout) -> float | torch.Tensor:
        assert not self.trainer.args.ema_model, "EMA is not supported in PMFEnv"
        model = self.get_eval_model()
        assert isinstance(model, DiTPMF)
        self.eval_rng.manual_seed(42)
        with self.autocast():
            outputs = sample(
                model,
                torch.randn((16, 3, IMAGE_SIZE, IMAGE_SIZE), generator=self.eval_rng, device=self.device),
                steps=self.sample_steps,
            )
        output_file = path.join(self.trainer.args.model_dir, f"sample_{self.trainer.epoch}.png")
        TF.to_pil_image(make_grid(outputs, nrow=8)).save(output_file)

        return super().eval_end(file=file)


class PMFTrainer(Trainer):
    def setup(self) -> None:
        assert not self.args.ema_model, "EMA is not supported in PMFTrainer"

    def create_dataloader(self, type: str) -> torch.utils.data.DataLoader:
        assert type in {"train", "eval"}
        if type == "train":
            dataset = CelebADataset(self.args.data_dir, split="train")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True,
            )
        else:
            dataset = CelebADataset(self.args.data_dir, split="test")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True,
            )

    def create_model(self) -> DiTPMF:
        return DiTPMF(aux_head_depth=self.args.aux_head_depth).to(self.device)

    def create_env(self) -> PMFEnv:
        return PMFEnv(
            self.model,
            loss_type=self.args.loss,
            p_mean=self.args.p_mean,
            p_std=self.args.p_std,
            data_proportion=self.args.data_proportion,
            norm_p=self.args.norm_p,
            norm_eps=self.args.norm_eps,
            lpips_lambda=self.args.lpips_lambda,
            perceptual_max_t=self.args.perceptual_max_t,
            sample_steps=self.args.sample_steps,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=40000)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "char"])
    parser.add_argument("--aux-head-depth", type=int, default=4)
    parser.add_argument("--p-mean", type=float, default=0.8)
    parser.add_argument("--p-std", type=float, default=0.8)
    parser.add_argument("--data-proportion", type=float, default=0.5)
    parser.add_argument("--norm-p", type=float, default=1.0)
    parser.add_argument("--norm-eps", type=float, default=0.01)
    parser.add_argument("--lpips-lambda", type=float, default=0.4)
    parser.add_argument("--perceptual-max-t", type=float, default=0.8)
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.set_defaults(
        batch_size=32,
        num_workers=8,
        max_epoch=400,
        learning_rate=0.001,
        optimizer="adamw_schedulefree",
        warmup_epoch=1,
        eval_step=4,
        amp_float="bfloat16",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    trainer = PMFTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
