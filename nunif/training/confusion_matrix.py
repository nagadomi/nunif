import torch
import sys


class SoftmaxConfusionMatrix():
    def __init__(self, class_names, max_print_class=None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.max_print_class = max_print_class
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long)

    def update(self, z, y):
        for t, p in zip(y, z):
            self.confusion_matrix[t.long(), p.long()] += 1

    def matrix(self):
        return self.confusion_matrix

    def class_accuracy(self):
        return self.confusion_matrix.diag() / (self.confusion_matrix.sum(1) + 1e-6)

    def average_row_correct(self):
        return self.class_accuracy().mean()

    def global_correct(self):
        return self.confusion_matrix.diag().sum() / (self.confusion_matrix.sum() + 1e-6)

    def clear(self):
        self.confusion_matrix.zero_()

    def print(self, file=sys.stdout):
        print(f" global correct:      {self.global_correct()},"
              f" average_row_correct: {self.average_row_correct()}",
              file=file)
        if self.max_print_class is not None and self.max_print_class < 0:
            return

        print(self.confusion_matrix, file=file)
        class_accuracy = self.class_accuracy()
        for i, name in enumerate(self.class_names):
            if self.max_print_class is not None and self.max_print_class <= i:
                print("...")
                break
            print(f"  {name}: {round(class_accuracy[i].item(), 4)}")
