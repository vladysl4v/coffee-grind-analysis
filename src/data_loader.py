"""
Coffee grind fineness dataset loader.

Usage
-----
from data_loader import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=32)

for images, labels in train_loader:
    # images: FloatTensor [B, 3, H, W], normalised to ImageNet stats
    # labels: FloatTensor [B]  (fineness value)
    ...
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images"
_LABELS_DIR = _ROOT / "data" / "labels"

_CSV = {
    "train": _LABELS_DIR / "labels_train.csv",
    "val":   _LABELS_DIR / "labels_val.csv",
    "test":  _LABELS_DIR / "labels_test.csv",
}

# ImageNet normalisation — sensible default for pretrained backbones
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    _NORMALIZE,
])

DEFAULT_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


class CoffeeDataset(Dataset):
    """Single-split dataset for coffee grind fineness regression.

    Parameters
    ----------
    split:     "train" | "val" | "test"
    transform: torchvision transform applied to each PIL image.
               Pass None to get raw PIL images.
    """

    def __init__(self, split: str, transform=None):
        if split not in _CSV:
            raise ValueError(f"split must be one of {list(_CSV)}, got {split!r}")

        csv_path = _CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found — run src/split_labels.py first."
            )

        df = pd.read_csv(csv_path, sep=";", decimal=",")
        self.samples   = df.iloc[:, 0].tolist()          # filenames
        self.labels    = df.iloc[:, 1].astype(float).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = _IMAGES_DIR / self.samples[idx]
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Place all images in {_IMAGES_DIR}"
            )
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    eval_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    batch_size:       images per batch
    num_workers:      DataLoader worker processes
    train_transform:  override the default augmentation pipeline
    eval_transform:   override the default eval pipeline (used for val + test)
    """
    t_train = train_transform or DEFAULT_TRAIN_TRANSFORM
    t_eval  = eval_transform  or DEFAULT_EVAL_TRANSFORM

    train_loader = DataLoader(
        CoffeeDataset("train", transform=t_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CoffeeDataset("val", transform=t_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CoffeeDataset("test", transform=t_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
