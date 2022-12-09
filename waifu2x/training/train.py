import os
import argparse
from os import path
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from .. import models  # register
from . dataset import Waifu2xScale2xDataset
from nunif.training.env import LuminancePSNREnv
from nunif.models import create_model, save_model, load_model, get_model_config
from nunif.modules import ClampLoss, LuminanceWeightedLoss, PSNR, AuxiliaryLoss, LBPLoss


def create_best_model_filename(args):
    if args.method == "scale":
        return path.join(args.model_dir, "scale2x.pth")
    else:
        raise NotImplementedError()


def create_checkpoint_filename(args):
    if args.method == "scale":
        return path.join(args.model_dir, "scale2x.checkpoint.pth")
    else:
        raise NotImplementedError()


def create_model_by_arch(args, device):
    kwargs = {"in_channels": 3, "out_channels": 3}
    if args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
        kwargs["no_clip"] = True

    model = create_model(args.arch, **kwargs)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model = model.to(device)
    return model


def create_dataloader(args, model):
    model_offset = get_model_config(model, "i2i_offset")
    if args.method == "scale":
        train_dataset = Waifu2xScale2xDataset(
            input_dir=path.join(args.data_dir, "train"),
            model_offset=model_offset,
            tile_size=args.size,
            num_samples=args.num_samples
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.minibatch_size,
            worker_init_fn=train_dataset.worker_init,
            shuffle=False,
            pin_memory=True,
            sampler=train_dataset.sampler(),
            num_workers=args.num_workers,
            drop_last=True)

        validation_dataset = Waifu2xScale2xDataset(
            input_dir=path.join(args.data_dir, "validation"),
            model_offset=model_offset,
            tile_size=args.size, validation=True)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.minibatch_size,
            worker_init_fn=validation_dataset.worker_init,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

        return train_loader, validation_loader
    else:
        raise NotImplementedError()


class Waifu2xEnv(LuminancePSNREnv):
    pass


def train(args):
    if args.size % 4 != 0:
        raise ValueError("--size must be a multiple of 4")
    if args.gpu[0] >= 0:
        device = f"cuda:{args.gpu[0]}"
    else:
        device = "cpu"
    best_model_filename = create_best_model_filename(args)
    checkpoint_filename = create_checkpoint_filename(args)
    if args.resume:
        model, meta = load_model(checkpoint_filename, device_ids=args.gpu)
    else:
        model = create_model_by_arch(args, device)
        meta = {}
    train_loader, validation_loader = create_dataloader(args, model)
    if args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
        criterion = AuxiliaryLoss([
            ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
            ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1)))],
            weight=(1.0, 0.5)).to(device)
    else:
        criterion = ClampLoss(LuminanceWeightedLoss(nn.HuberLoss(delta=0.3)))

    env = Waifu2xEnv(model, criterion=criterion)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer,
                       step_size=args.learning_rate_decay_step[0],
                       gamma=args.learning_rate_decay)
    grad_scaler = torch.cuda.amp.GradScaler()
    best_loss = 100000
    start_epoch = 1

    os.makedirs(args.model_dir, exist_ok=True)

    if not (args.disable_amp or device == "cpu"):
        env.enable_amp()

    if args.resume:
        if not args.reset_state:
            optimizer.load_state_dict(meta["optimizer_state_dict"])
            scheduler.load_state_dict(meta["scheduler_state_dict"])
            grad_scaler.load_state_dict(meta["grad_scaler_state_dict"])
            start_epoch = meta["last_epoch"] + 1
            best_loss = meta["best_loss"]
            print(f"** resume from epoch={meta['last_epoch']}, best_loss={best_loss}")
        else:
            print(f"** resume only model weight from {checkpoint_filename}")

    for epoch in range(start_epoch, args.max_epoch):
        print(f"* epoch: {epoch}, lr: {scheduler.get_last_lr()}")
        print("** train")

        env.train(
            loader=train_loader,
            optimizer=optimizer,
            grad_scaler=grad_scaler)
        scheduler.step()

        print("** validation")
        loss = env.validate(validation_loader)
        if loss < best_loss:
            print("+++ model updated")
            best_loss = loss
            save_model(model, best_model_filename, train_kwargs=args)

        save_model(
            model,
            checkpoint_filename,
            train_kwargs=args,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            grad_scaler_state_dict=grad_scaler.state_dict(),
            best_loss=best_loss,
            last_epoch=epoch)


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, choices=["scale"], required=True, help="waifu2x method")
    parser.add_argument("--arch", type=str,
                        choices=["waifu2x.cunet", "waifu2x.upcunet", "waifu2x.upconv_7", "waifu2x.vgg_7"],
                        required=True, help="network arch")
    parser.add_argument("--size", type=int, default=104, help="input size")
    parser.add_argument("--num-samples", type=int, default=50000, help="number of samples for each epoch")

    parser.set_defaults(minibatch_size=8)
    parser.set_defaults(learning_rate=0.0002)
    parser.set_defaults(learning_rate_decay=0.995)
    parser.set_defaults(learning_rate_decay_step=[1])

    parser.set_defaults(handler=train)

    return parser
