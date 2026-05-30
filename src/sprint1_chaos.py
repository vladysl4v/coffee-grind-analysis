"""
Replicates Sprint 1 simple_cnn training WITHOUT label scaling.
Labels are kept as raw 0-100 values — this causes the MSE to explode
and training to oscillate, exactly as seen in the Sprint 1 loss curve.

Run: uv run python src/sprint1_chaos.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.simple_cnn import SimpleCNN

_ROOT = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

_CSV = {
    "train": _LABELS_DIR / "train.csv",
    "val":   _LABELS_DIR / "val.csv",
}


class UnscaledCoffeeDataset(Dataset):
    """Labels are raw 0-100 — NO division by 100. This is the Sprint 1 bug."""

    def __init__(self, split: str):
        csv_path = _CSV[split]
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        self.samples = df.iloc[:, 0].tolist()
        self.labels  = df.iloc[:, 1].astype(float).tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = _IMAGES_DIR / self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = _TRANSFORM(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # raw, no /100
        return image, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Labels: RAW 0-100 (no scaling) — expect chaos\n")

    train_loader = DataLoader(UnscaledCoffeeDataset("train"), batch_size=32, shuffle=True)
    val_loader   = DataLoader(UnscaledCoffeeDataset("val"),   batch_size=32, shuffle=False)

    model     = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 31):
        model.train()
        train_mse = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss  = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse += loss.item()
        train_mse /= len(train_loader)

        model.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds   = model(images)
                val_mse += criterion(preds, labels).item()
                val_mae += (preds - labels).abs().mean().item()
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        print(f"Epoch {epoch:2d}/30 | train_mse={train_mse:8.2f} | val_mse={val_mse:8.2f} | val_mae={val_mae:6.2f}")


if __name__ == "__main__":
    main()
