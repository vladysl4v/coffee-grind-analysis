"""
Coffee grind fineness dataset loader.

Usage
-----
from data_loader import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=32)

for images, labels in train_loader:
    # images: FloatTensor [B, 3, H, W], normalised with dataset-specific stats
    # labels: FloatTensor [B]  (fineness value in [0, 1])
    ...
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from image_ops import MaskBoundingBoxCrop

_ROOT            = Path(__file__).parent.parent
_IMAGES_DIR      = _ROOT / "data" / "images" / "segmentation"
_RAW_IMAGES_DIR  = _ROOT / "data" / "images" / "raw"
_AUG_IMAGES_DIR  = _ROOT / "data" / "images" / "augmented_segmentation"
_RAW_AUG_IMAGES_DIR = _ROOT / "data" / "images" / "augmented_raw"
_LABELS_DIR      = _ROOT / "data" / "labels"
_STATS_PATH      = _ROOT / "data" / "dataset_stats.json"

# Default train pipeline: mask-aware crop + dataset normalisation only.
# Stochastic augmentations for ``--online-augment`` live in ``augmentation.transforms``.

_CSV = {
    "train": _LABELS_DIR / "train.csv",
    "val":   _LABELS_DIR / "val.csv",
    "test":  _LABELS_DIR / "test.csv",
}

_AUG_CSV = {
    "train": _LABELS_DIR / "augmented_train.csv",
    "val":   _LABELS_DIR / "val.csv",
    "test":  _LABELS_DIR / "test.csv",
}

_RAW_AUG_CSV = {
    "train": _LABELS_DIR / "augmented_raw_train.csv",
    "val":   _LABELS_DIR / "val.csv",
    "test":  _LABELS_DIR / "test.csv",
}

# Fallback stats (CenterCrop baseline); prefer values from data/dataset_stats.json
_DEFAULT_MEAN = [0.1557, 0.0899, 0.0404]
_DEFAULT_STD = [0.0483, 0.0349, 0.0190]


def _load_dataset_stats(source: str = "segmentation") -> tuple[list[float], list[float]]:
    if not _STATS_PATH.is_file():
        return _DEFAULT_MEAN, _DEFAULT_STD
    with _STATS_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    block = payload.get(source)
    if isinstance(block, dict):
        return list(block["mean"]), list(block["std"])
    # Legacy single-block file
    if "mean" in payload and "std" in payload:
        return list(payload["mean"]), list(payload["std"])
    return _DEFAULT_MEAN, _DEFAULT_STD


def build_normalize(
    *,
    use_augmented_data: bool = False,
    use_augmented_raw: bool = False,
    use_raw: bool = False,
) -> transforms.Normalize:
    if use_augmented_raw:
        source = "raw"
    elif use_augmented_data:
        source = "augmented_segmentation"
    elif use_raw:
        source = "raw"
    else:
        source = "segmentation"
    mean, std = _load_dataset_stats(source)
    return transforms.Normalize(mean=mean, std=std)


_DATASET_MEAN, _DATASET_STD = _load_dataset_stats("segmentation")

_NORMALIZE = transforms.Normalize(mean=_DATASET_MEAN, std=_DATASET_STD)

MASK_CROP = MaskBoundingBoxCrop(size=224, padding_ratio=0.05, threshold=8)

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    MASK_CROP,
    transforms.ToTensor(),
    _NORMALIZE,
])

DEFAULT_EVAL_TRANSFORM = transforms.Compose([
    MASK_CROP,
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

    def __init__(self, split: str, transform=None, csv_map=None, images_dir=None):
        csv_map = csv_map or _CSV
        if split not in csv_map:
            raise ValueError(f"split must be one of {list(csv_map)}, got {split!r}")

        self._images_dir = images_dir or _IMAGES_DIR
        csv_path = csv_map[split]
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
        img_path = self._images_dir / self.samples[idx]
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Expected images under {self._images_dir} (see README: raw vs segmentation vs augmented paths)."
            )
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx] / 100.0, dtype=torch.float32)
        return image, label


def get_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    eval_transform=None,
    use_augmented_data: bool = False,
    use_augmented_raw: bool = False,
    use_raw: bool = False,
    worker_init_fn=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    batch_size:       images per batch
    num_workers:      DataLoader worker processes
    train_transform:  override the default augmentation pipeline
    eval_transform:   override the default eval pipeline (used for val + test)
    """
    if use_augmented_raw:
        csv_map = _RAW_AUG_CSV
        img_dir = _RAW_AUG_IMAGES_DIR
    elif use_augmented_data:
        csv_map = _AUG_CSV
        img_dir = _AUG_IMAGES_DIR
    elif use_raw:
        csv_map = _CSV
        img_dir = _RAW_IMAGES_DIR
    else:
        csv_map = _CSV
        img_dir = _IMAGES_DIR

    normalize = build_normalize(
        use_augmented_data=use_augmented_data,
        use_augmented_raw=use_augmented_raw,
        use_raw=use_raw,
    )
    default_transform = transforms.Compose([MASK_CROP, transforms.ToTensor(), normalize])
    t_train = train_transform or default_transform
    t_eval = eval_transform or default_transform

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    train_loader = DataLoader(
        CoffeeDataset("train", transform=t_train, csv_map=csv_map, images_dir=img_dir),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        CoffeeDataset("val", transform=t_eval, csv_map=csv_map, images_dir=img_dir),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    test_loader = DataLoader(
        CoffeeDataset("test", transform=t_eval, csv_map=csv_map, images_dir=img_dir),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    return train_loader, val_loader, test_loader
