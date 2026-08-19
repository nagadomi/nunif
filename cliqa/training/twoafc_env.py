import torch.nn as nn

from nunif.training.env import BaseEnv, get_model_device


class TwoAFCEnv(BaseEnv):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_model_device(self.model)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        assert hasattr(self.model, "dist2logit")

    def clear_loss(self):
        self.sum_loss = 0
        self.sum_acc = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.criterion.train()
        self.clear_loss()

    def train_step(self, data):
        ref, p0, p1, judge, *_ = data
        ref, p0, p1, judge = self.to_device(ref), self.to_device(p0), self.to_device(p1), self.to_device(judge)
        with self.autocast():
            d0 = self.model(p0, ref)
            d1 = self.model(p1, ref)
            logits = self.model.dist2logit(d0, d1)
            loss = self.criterion(logits, judge)
            acc = self.compute_accuracy(d0, d1, judge)
        self.sum_loss += loss.detach()
        self.sum_acc += acc.detach()
        self.sum_step += 1

        return loss

    def train_end(self):
        loss = self.to_scalar(self.sum_loss / self.sum_step)
        acc = self.to_scalar(self.sum_acc / self.sum_step)
        if hasattr(self.model, "sparsity"):
            sparsity = self.model.sparsity()
        else:
            sparsity = None

        print(f"loss: {loss}, acc: {acc}, sparsity: {sparsity}")
        return loss

    def eval_begin(self):
        model = self.get_eval_model()
        model.eval()
        self.clear_loss()

    @staticmethod
    def compute_accuracy(d0, d1, judge):
        # ref: original reference patches
        # p0,p1: two distorted patches
        # judge: human judgments - 0 if all preferred p0, 1 if all humans preferred p1
        d1_lt_d0 = (d1 < d0).float()
        return (d1_lt_d0 * judge + (1 - d1_lt_d0) * (1 - judge)).mean()

    def eval_step(self, data):
        ref, p0, p1, judge, *_ = data
        ref, p0, p1, judge = self.to_device(ref), self.to_device(p0), self.to_device(p1), self.to_device(judge)
        model = self.get_eval_model()
        with self.autocast():
            d0 = model(p0, ref)
            d1 = model(p1, ref)
            acc = self.compute_accuracy(d0, d1, judge)
        self.sum_acc += acc.detach()
        self.sum_step += 1

        return -acc

    def eval_end(self):
        acc = self.to_scalar(self.sum_acc / self.sum_step)
        print(f"loss: {acc}")
        return -acc
