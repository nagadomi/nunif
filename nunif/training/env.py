import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from . confusion_matrix import SoftmaxConfusionMatrix
from .. models.utils import get_model_config, get_model_device
from .. modules import ClampLoss, LuminanceWeightedLoss, LuminancePSNR, PSNR
from .. device import autocast
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self):
        self.amp = False
        self.amp_dtype = torch.bfloat16
        self.trainer = None

    def enable_amp(self):
        self.amp = True

    def set_amp_dtype(self, dtype):
        self.amp_dtype = dtype

    def autocast(self, device=None):
        device = device or getattr(self, "device", None)
        assert device is not None

        return autocast(device, dtype=self.amp_dtype, enabled=self.amp)

    @abstractmethod
    def train_begin(self):
        pass

    @abstractmethod
    def train_step(self, data):
        pass

    def train_loss_hook(self, data, loss):
        pass

    def backward(self, loss, grad_scaler):
        losses = loss if isinstance(loss, (list, tuple)) else [loss]
        for loss in losses:
            if self.amp:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

    def optimizer_step(self, optimizer, grad_scaler):
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        if self.amp:
            for optimizer in optimizers:
                grad_scaler.step(optimizer)
                optimizer.zero_grad()
            grad_scaler.update()
        else:
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

    def train_backward_step(self, loss, optimizers, grad_scaler, update):
        self.backward(loss, grad_scaler)
        if update:
            self.optimizer_step(optimizers, grad_scaler)

    def calculate_adaptive_weight(self, base_loss, second_loss, param,
                                  grad_scaler, min=1e-6, max=1., mode="norm"):
        base_loss = grad_scaler.scale(base_loss)
        second_loss = grad_scaler.scale(second_loss)
        # ref. taming transformers
        base_grad = torch.autograd.grad(base_loss, param, retain_graph=True)[0]
        second_grad = torch.autograd.grad(second_loss, param, retain_graph=True)[0]
        assert base_grad is not None and second_grad is not None
        inv_scale = 1.0 / grad_scaler.get_scale()
        base_grad = base_grad * inv_scale
        second_grad = second_grad * inv_scale
        if mode == "norm":
            base_grad_strength = torch.norm(base_grad, p=2)
            second_grad_strength = torch.norm(second_grad, p=2) + 1e-6
        elif mode == "max":
            base_grad_strength = torch.max(torch.abs(base_grad))
            second_grad_strength = torch.max(torch.abs(second_grad)) + 1e-6
        else:
            raise NotImplementedError()
        grad_ratio = torch.clamp(base_grad_strength / second_grad_strength, min, max).item()
        if False:
            print("base", base_grad_strength.item(), "second", second_grad_strength.item(),
                  "weight", (base_grad_strength / second_grad_strength).item(),
                  "inv_scale", inv_scale)
        return grad_ratio

    @abstractmethod
    def train_end(self):
        pass

    @abstractmethod
    def eval_begin(self):
        pass

    @abstractmethod
    def eval_step(self, data):
        pass

    @abstractmethod
    def eval_end(self):
        loss = 0
        return loss

    def to_device(self, input, device=None):
        device = device or getattr(self, "device")
        if torch.is_tensor(input):
            return input.to(device)
        if isinstance(input, (tuple, list)):
            new_list = [self.to_device(elm, device) for elm in input]
            return tuple(new_list) if isinstance(input, tuple) else new_list
        # unknown type
        return input

    @staticmethod
    def check_nan(loss):
        losses = loss if isinstance(loss, (list, tuple)) else [loss]
        for loss in (losses):
            if torch.is_tensor(loss) and torch.isnan(loss).any().item():
                raise FloatingPointError("loss is NaN")

    def train(self, loader, optimizers, schedulers, grad_scaler, backward_step=1):
        assert backward_step > 0

        self.train_begin()
        for optimizer in optimizers:
            optimizer.zero_grad()
        t = 1
        for data in tqdm(loader, ncols=80):
            loss = self.train_step(data)
            if backward_step > 1:
                if isinstance(loss, (list, tuple)):
                    loss = [ls / backward_step for ls in loss]
                else:
                    loss = loss / backward_step
            self.train_loss_hook(data, loss)
            self.check_nan(loss)
            self.train_backward_step(loss, optimizers, grad_scaler,
                                     update=t % backward_step == 0)
            t += 1
        for scheduler in schedulers:
            scheduler.step()

        self.train_end()

    def eval(self, loader):
        self.eval_begin()
        if loader is not None:
            for data in tqdm(loader, ncols=80):
                with torch.no_grad():
                    self.eval_step(data)
        return self.eval_end()


class SoftmaxEnv(BaseEnv):
    def __init__(self, model, criterion=None, eval_tta=False, max_print_class=16):
        super().__init__()
        self.eval_tta = eval_tta
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.NLLLoss().to(self.device)
        self.class_names = get_model_config(model, "softmax_class_names")
        self.confusion_matrix = SoftmaxConfusionMatrix(self.class_names, max_print_class=max_print_class)

    def train_begin(self):
        self.model.train()
        self.confusion_matrix.clear()

    def train_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        with self.autocast():
            z = self.model(x)
            loss = self.criterion(z, y)
        self.confusion_matrix.update(torch.argmax(z, dim=1).cpu(), y.cpu())
        return loss

    def train_end(self):
        self.confusion_matrix.print()
        return 1 - self.confusion_matrix.average_row_correct()

    def eval_begin(self):
        self.model.eval()
        self.confusion_matrix.clear()

    def eval_step(self, data):
        x, y, *_ = data
        if self.eval_tta:
            B, TTA, = x.shape[:2]
            x = self.to_device(x)
            x = x.reshape(B * TTA, *x.shape[2:])
            with self.autocast():
                z = self.model(x)
            z = z.reshape(B, TTA, *z.shape[1:]).mean(dim=1)
            self.confusion_matrix.update(torch.argmax(z, dim=1).cpu(), y)
        else:
            x = self.to_device(x)
            with self.autocast():
                z = self.model(x)
            self.confusion_matrix.update(torch.argmax(z, dim=1).cpu(), y)

    def eval_end(self):
        self.confusion_matrix.print()
        return 1 - self.confusion_matrix.average_row_correct()


class I2IEnv(BaseEnv):
    def __init__(self, model, criterion=None, eval_criterion=None):
        super().__init__()
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        self.eval_criterion = eval_criterion
        if self.criterion is None:
            self.criterion = ClampLoss(nn.HuberLoss()).to(self.device)
        else:
            self.criterion = self.criterion.to(self.device)
        if self.eval_criterion is None:
            self.eval_criterion = ClampLoss(nn.HuberLoss()).to(self.device)
        else:
            self.eval_criterion = self.eval_criterion.to(self.device)

    def clear_loss(self):
        self.sum_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        with self.autocast():
            z = self.model(x)
            loss = self.criterion(z, y)
        self.sum_loss += loss.item()
        self.sum_step += 1
        return loss

    def train_end(self):
        mean_loss = self.sum_loss / self.sum_step
        print(f"loss: {mean_loss}")
        return mean_loss

    def eval_begin(self):
        self.model.eval()
        self.clear_loss()

    def eval_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        with self.autocast():
            z = self.model(x)
            loss = self.eval_criterion(z, y)
        self.sum_loss += loss.item()
        self.sum_step += 1

    def print_eval_result(self, loss, file=sys.stdout):
        print(f"loss: {loss}", file=file)

    def eval_end(self, file=sys.stdout):
        mean_loss = self.sum_loss / self.sum_step
        self.print_eval_result(mean_loss, file=file)
        return mean_loss


class RGBPSNREnv(I2IEnv):
    def __init__(self, model, criterion=None):
        if criterion is None:
            criterion = ClampLoss(nn.HuberLoss(0.3))
        super().__init__(model, criterion=criterion, eval_criterion=PSNR())

    def print_eval_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch RGB-PSNR: {psnr}", file=file)


class LuminancePSNREnv(I2IEnv):
    def __init__(self, model, criterion=None):
        if criterion is None:
            criterion = ClampLoss(LuminanceWeightedLoss(nn.HuberLoss(0.3)))
        super().__init__(model, criterion=criterion, eval_criterion=LuminancePSNR())

    def print_eval_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch Y-PSNR: {psnr}", file=file)


class UnsupervisedEnv(BaseEnv):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        self.t = 0

    def clear_loss(self):
        self.sum_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        if isinstance(data, (tuple, list)):
            x, *_ = data
        else:
            x = data
        x = self.to_device(x)
        with self.autocast():
            z = self.model(x)
            loss = self.criterion(z)
        self.sum_loss += loss.item()
        self.sum_step += 1
        return loss

    def train_end(self):
        mean_loss = self.sum_loss / self.sum_step
        print(f"loss: {mean_loss}")
        return mean_loss

    def eval_begin(self):
        self.model.eval()

    def eval_step(self, data):
        pass

    def eval_end(self):
        return None
