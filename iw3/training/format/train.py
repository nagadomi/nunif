import argparse
import os
from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nunif.models import Model
from nunif.training.env import SoftmaxEnv
from nunif.training.trainer import Trainer
from nunif.utils.image_loader import ImageLoader, list_images
from nunif.utils.pil_io import load_image_simple

LABELS = ("FLAT", "SBS", "TB")


def extract_features(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    mid_w = w // 2
    mid_h = h // 2

    d_lr = float(np.mean(np.abs(arr[:, :mid_w] - arr[:, mid_w : mid_w * 2])))
    d_tb = float(np.mean(np.abs(arr[:mid_h, :] - arr[mid_h : mid_h * 2, :])))
    d_hflip = float(np.mean(np.abs(arr - arr[:, ::-1])))
    d_vflip = float(np.mean(np.abs(arr - arr[::-1, :])))

    return np.array([d_lr, d_tb, d_hflip, d_vflip], dtype=np.float32)


def extract_label_from_filename(filename: str) -> str | None:
    stem = path.splitext(path.basename(filename))[0]
    if stem.endswith("_SBS"):
        return "SBS"
    elif stem.endswith("_TB"):
        return "TB"
    elif stem.endswith("_FLAT"):
        return "FLAT"
    return None


def predict(
    im: Image.Image,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> str:
    x = extract_features(im)
    h_out = np.maximum(0.0, x @ w1.T + b1)
    out = h_out @ w2.T + b2
    return LABELS[int(np.argmax(out))]


class FormatDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.files: list[str] = []
        self.labels: list[int] = []

        image_files = list_images(data_dir)
        label_map = {"FLAT": 0, "SBS": 1, "TB": 2}
        for file_path in image_files:
            label_str = extract_label_from_filename(file_path)
            if label_str is not None:
                self.files.append(file_path)
                self.labels.append(label_map[label_str])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[index]
        label = self.labels[index]
        im, _ = load_image_simple(file_path, color="rgb")
        if im is None:
            feat = np.zeros(4, dtype=np.float32)
        else:
            feat = extract_features(im)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


class FormatClassifier(Model):
    name = "iw3.format_classifier"

    def __init__(self, in_features: int = 4, hidden_size: int = 4, num_classes: int = 3):
        super().__init__(locals())
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax_class_names = list(LABELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return F.log_softmax(out, dim=-1)


def evaluate_predict(
    eval_dir: str,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> None:
    image_files = list_images(eval_dir)
    if not image_files:
        print(f"No images found in {eval_dir}")
        return

    loader = ImageLoader(
        files=image_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"},
    )

    cm = {true_lbl: {pred_lbl: 0 for pred_lbl in LABELS} for true_lbl in LABELS}
    total = 0
    correct = 0

    for item in tqdm(loader, desc="Eval predict", ncols=80):
        im, meta = item
        if im is None:
            continue
        filename = meta["filename"]
        true_label = extract_label_from_filename(filename)
        if true_label is None:
            continue

        pred_label = predict(im, w1=w1, b1=b1, w2=w2, b2=b2)
        cm[true_label][pred_label] += 1
        total += 1
        if pred_label == true_label:
            correct += 1

    if total == 0:
        return

    print(f"\nNumPy predict result on eval ({correct}/{total}, {correct / total * 100:.2f}%):")
    headers = ["True \\ Pred"] + list(LABELS) + ["Accuracy"]
    print(f"{headers[0]:<14} " + " ".join(f"{h:>8}" for h in headers[1:]))
    print("-" * (14 + len(headers[1:]) * 9))
    for true_label in LABELS:
        row_counts = [cm[true_label][pred_label] for pred_label in LABELS]
        row_total = sum(row_counts)
        row_correct = cm[true_label][true_label]
        acc_str = f"{row_correct / row_total * 100:6.2f}%" if row_total > 0 else "  N/A  "
        print(f"{true_label:<14} " + " ".join(f"{c:>8d}" for c in row_counts) + f" {acc_str:>8}")


def export_text_function(model: FormatClassifier, output_file: str | None = None) -> str:
    fc1_w = model.fc1.weight.detach().cpu().numpy().tolist()
    fc1_b = model.fc1.bias.detach().cpu().numpy().tolist()
    fc2_w = model.fc2.weight.detach().cpu().numpy().tolist()
    fc2_b = model.fc2.bias.detach().cpu().numpy().tolist()

    code = f"""# Auto-generated format classification function (no PyTorch required)
import numpy as np
from PIL import Image

_W1 = np.array({fc1_w!r}, dtype=np.float32)
_B1 = np.array({fc1_b!r}, dtype=np.float32)
_W2 = np.array({fc2_w!r}, dtype=np.float32)
_B2 = np.array({fc2_b!r}, dtype=np.float32)
_LABELS = {list(LABELS)!r}


def extract_features(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    mid_w = w // 2
    mid_h = h // 2

    d_lr = float(np.mean(np.abs(arr[:, :mid_w] - arr[:, mid_w : mid_w * 2])))
    d_tb = float(np.mean(np.abs(arr[:mid_h, :] - arr[mid_h : mid_h * 2, :])))
    d_hflip = float(np.mean(np.abs(arr - arr[:, ::-1])))
    d_vflip = float(np.mean(np.abs(arr - arr[::-1, :])))

    return np.array([d_lr, d_tb, d_hflip, d_vflip], dtype=np.float32)


def predict(
    im: Image.Image,
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
) -> str:
    x = extract_features(im)
    _w1 = w1 if w1 is not None else _W1
    _b1 = b1 if b1 is not None else _B1
    _w2 = w2 if w2 is not None else _W2
    _b2 = b2 if b2 is not None else _B2

    # 2-layer MLP: ReLU(x @ W1.T + B1) @ W2.T + B2
    h_out = np.maximum(0.0, x @ _w1.T + _b1)
    out = h_out @ _w2.T + _b2
    return _LABELS[int(np.argmax(out))]
"""

    print("\n" + "=" * 25 + " Exported Predict Function " + "=" * 25)
    print(code)
    print("=" * 77 + "\n")

    if output_file:
        os.makedirs(path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Exported predict function saved to: {output_file}")

    return code


class FormatTrainer(Trainer):
    def create_model(self) -> Model:
        model = FormatClassifier(in_features=4, hidden_size=self.args.hidden_size, num_classes=len(LABELS))
        return model.to(self.device)

    def create_dataloader(self, type: str) -> DataLoader:
        dataset_dir = path.join(self.args.data_dir, type)
        dataset = FormatDataset(dataset_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(type == "train"),
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        return loader

    def create_env(self) -> SoftmaxEnv:
        return SoftmaxEnv(self.model, eval_tta=False)

    def fit(self) -> None:
        super().fit()
        model = self.ema_model.module if self.args.ema_model else self.model
        assert isinstance(model, FormatClassifier)

        w1 = model.fc1.weight.detach().cpu().numpy()
        b1 = model.fc1.bias.detach().cpu().numpy()
        w2 = model.fc2.weight.detach().cpu().numpy()
        b2 = model.fc2.bias.detach().cpu().numpy()

        eval_dir = path.join(self.args.data_dir, "eval")
        if path.exists(eval_dir):
            evaluate_predict(eval_dir, w1=w1, b1=b1, w2=w2, b2=b2)

        output_file = path.join(self.args.model_dir, "predict.py") if self.args.model_dir else None
        export_text_function(model, output_file=output_file)


def train(args: argparse.Namespace) -> None:
    trainer = FormatTrainer(args)
    trainer.fit()


def register(
    subparsers: argparse._SubParsersAction, default_parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "iw3.format", parents=[default_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hidden-size", type=int, default=4, help="hidden layer size for MLP")
    parser.set_defaults(
        handler=train,
        learning_rate=0.01,
        scheduler="cosine",
        learning_cycles=1,
        max_epoch=40,
        batch_size=32,
    )
    return parser
