import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from . confusion_matrix import SoftmaxConfusionMatrix
from .. models.utils import get_model_config, get_model_device
from .. modules import ClampLoss, LuminanceWeightedLoss, LuminancePSNR, PSNR

from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self):
        self.amp = False
        self.trainer = None

    def enable_amp(self):
        self.amp = True

    @abstractmethod
    def train_begin(self):
        pass

    @abstractmethod
    def train_step(self, data):
        pass

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

    def to_device(self, input, device):
        if torch.is_tensor(input):
            return input.to(device)
        if isinstance(input, (tuple, list)):
            new_list = [self.to_device(elm, device) for elm in input]
            return tuple(new_list) if isinstance(input, tuple) else new_list
        # unknown type
        return input

    def train(self, loader, optimizer, grad_scaler):
        self.train_begin()
        for data in tqdm(loader, ncols=80):
            optimizer.zero_grad()
            loss = self.train_step(data)
            if self.amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
        self.train_end()

    def eval(self, loader):
        self.eval_begin()
        if loader is not None:
            for data in tqdm(loader, ncols=80):
                with torch.no_grad():
                    self.eval_step(data)
        return self.eval_end()


class SoftmaxEnv(BaseEnv):
    def __init__(self, model, criterion=None, eval_tta=False):
        super().__init__()
        self.eval_tta = eval_tta
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.NLLLoss().to(self.device)
        self.class_names = get_model_config(model, "softmax_class_names")
        self.confusion_matrix = SoftmaxConfusionMatrix(self.class_names, max_print_class=16)

    def train_begin(self):
        self.model.train()
        self.confusion_matrix.clear()

    def train_step(self, data):
        x, y = data
        x, y = self.to_device(x, self.device), self.to_device(y, self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
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
        x, y = data
        if self.eval_tta:
            B, TTA, = x.shape[:2]
            x = x.to(self.device)
            x = x.reshape(B * TTA, *x.shape[2:])
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                z = self.model(x)
            z = z.reshape(B, TTA, *z.shape[1:]).mean(dim=1)
            self.confusion_matrix.update(torch.argmax(z, dim=1).cpu(), y)
        else:
            x = x.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
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
        x, y = data
        x, y = self.to_device(x, self.device), self.to_device(y, self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
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
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
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
        x = data
        x = self.to_device(x, self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
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
