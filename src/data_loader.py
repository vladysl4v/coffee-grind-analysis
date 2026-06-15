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

_ROOT            = Path(__file__).parent.parent
_IMAGES_DIR      = _ROOT / "data" / "images" / "segmentation"
_RAW_IMAGES_DIR  = _ROOT / "data" / "images" / "raw"
_AUG_IMAGES_DIR  = _ROOT / "data" / "images" / "augmented_segmentation"
_RAW_AUG_IMAGES_DIR = _ROOT / "data" / "images" / "augmented_raw"
_LABELS_DIR      = _ROOT / "data" / "labels"

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

# Dataset-specific normalisation computed over the training set
_NORMALIZE = transforms.Normalize(
    mean=[0.1557, 0.0899, 0.0404],
    std =[0.0483, 0.0349, 0.0190],
)

from torchvision.transforms import functional as F

class TwoOffsetCenterCrops:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.crop_h, self.crop_w = size

    def __call__(self, img):
        width, height = F.get_image_size(img)

        center_x = width // 2
        center_y = height // 2

        top = center_y - self.crop_h // 2

        # left crop: right boundary at center
        left1 = center_x - self.crop_w
        crop1 = F.crop(img, top, left1, self.crop_h, self.crop_w)

        # right crop: left boundary at center
        left2 = center_x
        crop2 = F.crop(img, top, left2, self.crop_h, self.crop_w)

        return crop1, crop2

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    _NORMALIZE,
    TwoOffsetCenterCrops(224),
])

DEFAULT_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    _NORMALIZE,
    TwoOffsetCenterCrops(224),
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
        return len(self.samples) * 2

    def __getitem__(self, idx: int):
        img_path = self._images_dir / self.samples[idx // 2]
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Place all images in {_IMAGES_DIR}"
            )
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img1, img2 = self.transform(image)
        if idx % 2 == 0:
            image = img1
        else:
            image = img2
        label = torch.tensor(self.labels[idx // 2] / 100.0, dtype=torch.float32)
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
    t_train = train_transform or DEFAULT_TRAIN_TRANSFORM
    t_eval  = eval_transform  or DEFAULT_EVAL_TRANSFORM
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
