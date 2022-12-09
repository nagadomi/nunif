import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from . confusion_matrix import SoftMaxConfusionMatrix
from .. models.utils import get_model_config, get_model_device
from .. modules import ClampLoss, LuminanceWeightedLoss, LuminancePSNR, PSNR

from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self):
        self.amp = False

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
    def validation_begin(self):
        pass

    @abstractmethod
    def validation_step(self, data):
        pass

    @abstractmethod
    def validation_end(self):
        loss = 0
        return loss

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

    def validate(self, loader):
        self.validation_begin()
        for data in tqdm(loader, ncols=80):
            with torch.no_grad():
                self.validation_step(data)
        return self.validation_end()


class SoftMaxEnv(BaseEnv):
    def __init__(self, model, criterion=None):
        super().__init__()
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.class_names = get_model_config(model, "softmax_class_names")
        self.confusion_matrix = SoftMaxConfusionMatrix(self.class_names, max_print_class=16)

    def train_begin(self):
        self.model.train()
        self.confusion_matrix.clear()

    def train_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            z = self.model(x)
            loss = self.criterion(z, y)
        with torch.no_grad():
            self.confusion_matrix.update(torch.max(z, dim=1), y)
        return loss

    def train_end(self):
        self.consusion_matrix.print()
        return 1 - self.consusion_matrix.average_row_correct()

    def validation_begin(self):
        self.model.eval()
        self.confusion_matrix.clear()

    def validation_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            z = self.model(x)
        self.confusion_matrix.update(torch.max(z, dim=1), y)

    def validation_end(self):
        self.consusion_matrix.print()
        return 1 - self.consusion_matrix.average_row_correct()


class I2IEnv(BaseEnv):
    def __init__(self, model, criterion=None, validation_criterion=None):
        super().__init__()
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = criterion
        self.validation_criterion = validation_criterion
        if self.criterion is None:
            self.criterion = ClampLoss(nn.HuberLoss()).to(self.device)
        else:
            self.criterion = self.criterion.to(self.device)
        if self.validation_criterion is None:
            self.validation_criterion = ClampLoss(nn.HuberLoss()).to(self.device)
        else:
            self.validation_criterion = self.validation_criterion.to(self.device)

    def clear_loss(self):
        self.sum_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
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

    def validation_begin(self):
        self.model.eval()
        self.clear_loss()

    def validation_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            z = self.model(x)
            loss = self.validation_criterion(z, y)
        self.sum_loss += loss.item()
        self.sum_step += 1

    def print_validation_result(self, loss, file=sys.stdout):
        print(f"loss: {loss}", file=file)

    def validation_end(self, file=sys.stdout):
        mean_loss = self.sum_loss / self.sum_step
        self.print_validation_result(mean_loss, file=file)
        return mean_loss


class RGBPSNREnv(I2IEnv):
    def __init__(self, model, criterion=None):
        if criterion is None:
            criterion = ClampLoss(nn.HuberLoss(0.3))
        super().__init__(model, criterion=criterion, validation_criterion=PSNR())

    def print_validation_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch RGB-PSNR: {psnr}", file=file)


class LuminancePSNREnv(I2IEnv):
    def __init__(self, model, criterion=None):
        if criterion is None:
            criterion = ClampLoss(LuminanceWeightedLoss(nn.HuberLoss(0.3)))
        super().__init__(model, criterion=criterion, validation_criterion=LuminancePSNR())

    def print_validation_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch Y-PSNR: {psnr}", file=file)
