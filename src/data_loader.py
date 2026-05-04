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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from numerical_features import (
    FEATURE_NAMES,
    extract_numerical_features,
    get_normalized_gray_crop,
)

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "segmentation"
_LABELS_DIR = _ROOT / "data" / "labels"
_FEATURES_DIR = _ROOT / "data" / "features" / "numerical"

_CSV = {
    "train": _LABELS_DIR / "train.csv",
    "val":   _LABELS_DIR / "val.csv",
    "test":  _LABELS_DIR / "test.csv",
}

# ImageNet normalisation — sensible default for pretrained backbones
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    _NORMALIZE,
])

DEFAULT_EVAL_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

DEFAULT_FEATURE_MODEL_TRANSFORM = transforms.Compose([
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

        label = torch.tensor(self.labels[idx] / 100.0, dtype=torch.float32)
        return image, label


class NumericalFeatureCoffeeDataset(CoffeeDataset):
    def __init__(
        self,
        split: str,
        image_mode: str = "rgb",
        transform=None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        super().__init__(split, transform=None)

        if image_mode not in {"rgb", "gray"}:
            raise ValueError(f"image_mode must be 'rgb' or 'gray', got {image_mode!r}")

        self.image_mode = image_mode
        self.image_transform = transform
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self._feature_cache: dict[int, np.ndarray] = {}
        self._precomputed_features = self._load_precomputed_features(split)

    def _load_image(self, idx: int) -> Image.Image:
        img_path = _IMAGES_DIR / self.samples[idx]
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Place all images in {_IMAGES_DIR}"
            )
        return Image.open(img_path).convert("RGB")

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.image_mode == "gray":
            gray = get_normalized_gray_crop(image)
            return torch.from_numpy(gray).unsqueeze(0).float() / 255.0

        transform = self.image_transform or DEFAULT_FEATURE_MODEL_TRANSFORM
        return transform(image)

    def _load_precomputed_features(self, split: str) -> dict[str, np.ndarray]:
        csv_path = _FEATURES_DIR / f"{split}_features.csv"
        if not csv_path.exists():
            return {}

        df = pd.read_csv(csv_path)
        required_columns = {"sample", *FEATURE_NAMES}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Precomputed feature CSV {csv_path} is missing columns: {sorted(missing_columns)}"
            )

        feature_map: dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            feature_map[row["sample"]] = row[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)

        missing_samples = sorted(set(self.samples).difference(feature_map))
        if missing_samples:
            raise ValueError(
                f"Precomputed feature CSV {csv_path} is missing {len(missing_samples)} samples "
                f"(for example: {missing_samples[:3]})"
            )

        return feature_map

    def _feature_vector(self, idx: int, image: Image.Image | None = None) -> np.ndarray:
        if idx not in self._feature_cache:
            sample = self.samples[idx]
            if sample in self._precomputed_features:
                self._feature_cache[idx] = self._precomputed_features[sample]
            else:
                if image is None:
                    image = self._load_image(idx)
                self._feature_cache[idx] = extract_numerical_features(image).astype(np.float32)
        return self._feature_cache[idx]

    def compute_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        features = np.stack(
            [self._feature_vector(idx) for idx in range(len(self))],
            axis=0,
        ).astype(np.float32)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def __getitem__(self, idx: int):
        image = self._load_image(idx)
        image_tensor = self._image_tensor(image)

        features = self._feature_vector(idx, image=image).copy()
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        label = torch.tensor(self.labels[idx] / 100.0, dtype=torch.float32)
        feature_tensor = torch.from_numpy(features.astype(np.float32))
        return image_tensor, feature_tensor, label


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

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(
        CoffeeDataset("train", transform=t_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        CoffeeDataset("val", transform=t_eval),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    test_loader = DataLoader(
        CoffeeDataset("test", transform=t_eval),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def get_numerical_feature_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_mode: str = "rgb",
    train_transform=None,
    eval_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    t_train = train_transform or DEFAULT_FEATURE_MODEL_TRANSFORM
    t_eval = eval_transform or DEFAULT_FEATURE_MODEL_TRANSFORM

    train_dataset = NumericalFeatureCoffeeDataset(
        "train",
        image_mode=image_mode,
        transform=t_train,
    )
    feature_mean, feature_std = train_dataset.compute_feature_stats()
    train_dataset.feature_mean = feature_mean
    train_dataset.feature_std = feature_std

    val_dataset = NumericalFeatureCoffeeDataset(
        "val",
        image_mode=image_mode,
        transform=t_eval,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_dataset = NumericalFeatureCoffeeDataset(
        "test",
        image_mode=image_mode,
        transform=t_eval,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    return train_loader, val_loader, test_loader
