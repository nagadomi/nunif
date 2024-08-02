import os
from os import path
import argparse
from multiprocessing import cpu_count
import torch
import torch.optim as optim
import torch.optim.swa_utils as swa_utils
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingWarmRestarts,
    ConstantLR, ChainedScheduler
)
from . cosine_wd import (
    CosineAnnealingWarmRestartsWithFixedWeightDecay,
    CosineAnnealingWarmRestartsWithScheduledWeightDecay,
)
from ..optim import Lion
from ..models import create_model, save_model, load_model
from ..initializer import set_seed
from ..device import create_device
from .weight_decay_config import configure_adamw
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class Trainer(ABC):
    def __init__(self, args):
        self.args = args
        self.initialized = False
        self.runtime_id = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d%H%M%S')

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.device = create_device(self.args.gpu)
        os.makedirs(self.args.model_dir, exist_ok=True)
        set_seed(self.args.seed)

        self.model = self.create_model()
        if self.args.checkpoint_file is not None:
            self.load_initial_parameters(self.args.checkpoint_file)
        self.setup_model()

        self.train_loader = self.create_dataloader(type="train")
        self.eval_loader = self.create_dataloader(type="eval")
        self.best_model_filename = self.create_best_model_filename()
        self.epoch = 1
        self.start_epoch = 1
        self.best_loss = 1000000000
        self.optimizers = self.create_optimizers()
        self.schedulers = self.create_schedulers(self.optimizers)
        self.grad_scaler = self.create_grad_scaler()
        if self.args.resume:
            self.resume()
        self.env = self.create_env()
        self.env.trainer = self

        if self.amp_is_enabled():
            self.env.enable_amp()
        self.env.set_amp_dtype(torch.bfloat16 if (self.args.amp_float == "bfloat16" or self.args.gpu[0] < 0) else torch.float16)
        self.log_fp = open(path.join(self.args.model_dir, f"loss_{self.runtime_id}.csv"), mode="w")

        if self.args.ema_model:
            self.ema_model = swa_utils.AveragedModel(
                self.model, multi_avg_fn=swa_utils.get_ema_multi_avg_fn(self.args.ema_decay))

        self.setup()

    def shutdown(self):
        if self.train_loader is not None:
            del self.train_loader
            self.train_loader = None

        if self.eval_loader is not None:
            del self.eval_loader
            self.eval_loader = None

        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None

    def setup(self):
        pass

    def setup_model(self):
        pass

    def amp_is_enabled(self):
        return not (self.args.disable_amp or self.device.type in {"cpu", "mps"})

    def resume(self):
        latest_checkpoint_filename = self.create_checkpoint_filename()
        _, meta = load_model(latest_checkpoint_filename, model=self.model)
        if not self.args.reset_state:
            if not isinstance(meta["optimizer_state_dict"], (list, tuple)):
                self.optimizers[0].load_state_dict(meta["optimizer_state_dict"])
            else:
                for i, optimizer_state_dict in enumerate(meta["optimizer_state_dict"]):
                    self.optimizers[i].load_state_dict(optimizer_state_dict)
            if not isinstance(meta["scheduler_state_dict"], (list, tuple)):
                self.schedulers[0].load_state_dict(meta["scheduler_state_dict"])
            else:
                for i, scheduler_state_dict in enumerate(meta["scheduler_state_dict"]):
                    self.schedulers[i].load_state_dict(scheduler_state_dict)
            self.grad_scaler.load_state_dict(meta["grad_scaler_state_dict"])
            self.start_epoch = meta["last_epoch"] + 1
            self.best_loss = meta["best_loss"]
        print(f"* load checkpoint from {latest_checkpoint_filename}")
        return meta

    def load_initial_parameters(self, checkpoint_filename):
        load_model(checkpoint_filename, model=self.model)

    @staticmethod
    def _lr_format(schedulers):
        lrs = []
        for scheduler in schedulers:
            lrs.append("[" + ", ".join([format(lr, '.3g') for lr in scheduler.get_last_lr()]) + "]")
        return "[" + ", ".join(lrs) + "]"

    def fit(self):
        self.initialize()
        try:
            for self.epoch in range(self.start_epoch, self.args.max_epoch + 1):
                print("-" * 64)
                print(f" epoch: {self.epoch}, lr: {self._lr_format(self.schedulers)}")
                print("--\n train")
                train_loss = self.env.train(
                    loader=self.train_loader,
                    optimizers=self.optimizers,
                    schedulers=self.schedulers,
                    grad_scaler=self.grad_scaler,
                    backward_step=self.args.backward_step,
                )
                if not self.args.skip_eval:
                    print("--\n eval")
                    loss = self.env.eval(self.eval_loader)
                    if loss is None:
                        self.save_best_model()
                    elif loss < self.best_loss:
                        print("* best model updated")
                        self.best_loss = loss
                        self.save_best_model()
                else:
                    self.save_best_model()
                self.save_checkpoint()
                try:
                    self.write_log(self.epoch, train_loss, loss)
                except:  # noqa
                    pass
        finally:
            self.shutdown()

    def create_model(self):
        if not hasattr(self.args, "arch"):
            raise NotImplementedError("--arch option is not implemented."
                                      " Add --arch option or override create_model()")
        return create_model(self.args.arch, device_ids=self.args.gpu)

    def create_optimizers(self):
        return [self.create_optimizer(self.model)]

    def create_optimizer(self, model, optimizer_type=None, lr=None, weight_decay=None, adam_beta1=None):
        optimizer_type = optimizer_type or self.args.optimizer
        lr = lr if lr is not None else self.args.learning_rate
        weight_decay = weight_decay if weight_decay is not None else self.args.weight_decay
        adam_beta1 = adam_beta1 if adam_beta1 is not None else self.args.adam_beta1

        if optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=lr,
                              betas=(adam_beta1, 0.999))
        elif optimizer_type == "adamw":
            return configure_adamw(
                model,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, 0.999))
        elif optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=self.args.momentum,
                weight_decay=weight_decay)
        elif optimizer_type == "lion":
            return Lion(model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"optimizer = {optimizer_type}")

    def create_schedulers(self, optimizers):
        return [self.create_scheduler(optimizer) for optimizer in optimizers]

    def create_scheduler(self, optimizer):
        # TODO: support more schedulers if needed
        if self.args.scheduler == "step":
            if len(self.args.learning_rate_decay_step) == 1:
                scheduler = StepLR(
                    optimizer,
                    step_size=self.args.learning_rate_decay_step[0],
                    gamma=self.args.learning_rate_decay)
            else:
                scheduler = MultiStepLR(
                    optimizer,
                    milestones=self.args.learning_rate_decay_step,
                    gamma=self.args.learning_rate_decay)
        elif self.args.scheduler in {"cosine", "cosine_wd", "cosine_fixed_wd"}:
            step = self.args.learning_rate_cycles
            t_0 = self.args.max_epoch // step
            old_max_epoch = self.args.max_epoch
            # Adjust epoch to keep the final epoch to the minimum LR
            self.args.max_epoch -= (self.args.max_epoch % step) + 1
            print(f"scheduler=cosine: max_epoch: {old_max_epoch} -> {self.args.max_epoch}")
            eta_min = self.args.learning_rate_cosine_min
            if self.args.scheduler == "cosine":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=t_0, eta_min=eta_min)
            elif self.args.scheduler == "cosine_wd":
                scheduler = CosineAnnealingWarmRestartsWithScheduledWeightDecay(
                    optimizer, T_0=t_0, eta_min=eta_min,
                    weight_decay_min=self.args.weight_decay,
                    weight_decay_max=self.args.weight_decay_end)
            elif self.args.scheduler == "cosine_fixed_wd":
                scheduler = CosineAnnealingWarmRestartsWithFixedWeightDecay(
                    optimizer, T_0=t_0, eta_min=eta_min)
        if self.args.warmup_epoch > 0:
            # TODO: `total_iters=self.args.warmup_epoch` does not work correctly,
            # ConstantLR works fine, but does not work correctly when used with ChainedScheduler.
            warmup_scheduler = ConstantLR(optimizer,
                                          factor=self.args.warmup_learning_rate / self.args.learning_rate,
                                          total_iters=self.args.warmup_epoch)
            scheduler = ChainedScheduler([warmup_scheduler, scheduler])

        return scheduler

    def create_grad_scaler(self):
        if hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(self.device.type, enabled=self.amp_is_enabled())
        else:
            return torch.cuda.amp.GradScaler(enabled=self.amp_is_enabled())

    def create_best_model_filename(self):
        return path.join(self.args.model_dir, f"{self.model.name}.pth")

    def create_checkpoint_filename(self):
        return path.join(self.args.model_dir, f"{self.model.name}.checkpoint.pth")

    def save_checkpoint(self, **kwargs):
        optimizer_state_dict = [optimizer.state_dict() for optimizer in self.optimizers]
        scheduler_state_dict = [scheduler.state_dict() for scheduler in self.schedulers]
        model = self.ema_model.module if self.args.ema_model else self.model
        save_model(
            model,
            self.create_checkpoint_filename(),
            train_kwargs=self.args,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            grad_scaler_state_dict=self.grad_scaler.state_dict(),
            best_loss=self.best_loss,
            last_epoch=self.epoch,
            **kwargs)

    def save_best_model(self):
        model = self.ema_model.module if self.args.ema_model else self.model
        save_model(model, self.best_model_filename)
        if not self.args.disable_backup:
            # backup file per runtime
            backup_file = f"{path.splitext(self.best_model_filename)[0]}.{self.runtime_id}.pth.bk"
            save_model(model, backup_file)

    def write_log(self, epoch, train_loss, eval_loss):
        def to_float(loss):
            if loss is None:
                return 0
            elif isinstance(loss, (list, tuple)):
                return sum([to_float(v) for v in loss]) / (len(loss) + 1e-6)
            elif torch.is_tensor(loss):
                return loss.detach().float().mean().item()
            else:
                try:
                    return float(loss)
                except ValueError:
                    return 0

        train_loss = to_float(train_loss)
        eval_loss = to_float(eval_loss)
        self.log_fp.write(f"{epoch},{train_loss},{eval_loss}\n")
        self.log_fp.flush()

    @abstractmethod
    def create_dataloader(self, type):
        assert (type in {"train", "eval"})

    @abstractmethod
    def create_env(self):
        pass


def create_trainer_default_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    num_workers = min(cpu_count() - 2, 8)
    if not num_workers > 0:
        num_workers = cpu_count()

    parser.add_argument("--data-dir", "-i", type=str, required=True,
                        help="input training data directory that created by `create_training_data` command")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="output directory for trained model/checkpoint")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="minibatch size")
    parser.add_argument("--backward-step", type=int, default=1,
                        help="number of times to accumulate gradient")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd", "lion"], default="adam",
                        help="optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="weight decay coefficient for adamw, sgd")
    parser.add_argument("--weight-decay-end", type=float, default=0.05,
                        help="max weight decay coefficient for cosine_wd")
    parser.add_argument("--adam-beta1", type=float, default=0.9,
                        help="beta1 hyperparameter for adam/adamw")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for sgd")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="number of worker processes for data loader")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="number of batches loaded in advance by each worker")
    parser.add_argument("--max-epoch", type=int, default=200,
                        help="max epoch")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0],
                        help="device ids; if -1 is specified, use CPU")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="learning rate")
    parser.add_argument("--scheduler", type=str, choices=["step", "cosine", "cosine_wd", "cosine_fixed_wd"], default="step",
                        help="learning rate scheduler")
    parser.add_argument("--learning-rate-decay", type=float, default=0.995,
                        help="learning rate decay for StepLR")
    parser.add_argument("--learning-rate-decay-step", type=int, nargs="+", default=[1],
                        help="learning rate decay step for StepLR/MultiStepLR")
    parser.add_argument("--learning-rate-cycles", type=int, default=5,
                        help="number of learning rate cycles for CosineAnnealingWarmRestarts")
    parser.add_argument("--learning-rate-cosine-min", type=float, default=1e-6,
                        help="Minimum learning rate for --schedule cosine")
    parser.add_argument("--warmup-epoch", type=int, default=0,
                        help="warmup epochs with --warmup-learning-rate")
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-6,
                        help="learning rate for warmup")
    parser.add_argument("--disable-amp", action="store_true",
                        help="disable AMP for some special reason")
    parser.add_argument("--amp-float", type=str, default="fp16", choices=["bfloat16", "fp16"],
                        help="dtype for autocast. bfloat16/fp16")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from the latest checkpoint file")
    parser.add_argument("--reset-state", action="store_true",
                        help="do not load best_score, optimizer and scheduler state when --resume")
    parser.add_argument("--seed", type=int, default=71,
                        help="random seed. if -1 is specified, a random number seed is used")
    parser.add_argument("--checkpoint-file", type=str,
                        help="checkpoint file for initializing model parameters. ignored when --resume is specified")
    parser.add_argument("--disable-backup", action="store_true",
                        help="disable backup of the best model file for every runtime")
    parser.add_argument("--ignore-nan", action="store_true",
                        help="do not raise NaN exception unless NaN occurs more than 100 times in one epoch")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval")
    parser.add_argument("--ema-model", action="store_true",
                        help="Use AveragedModel and save EMA model checkpoint")
    parser.add_argument("--ema-decay", type=float, default=0.98,
                        help="decay parameter for EMA model")
    parser.add_argument("--ema-step", type=int, default=8,
                        help="Update interval for EMA model")

    return parser
