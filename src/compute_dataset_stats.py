"""
Compute per-channel mean and std over the training set (pre-normalization).

Usage
-----
uv run python src/compute_dataset_stats.py
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CoffeeDataset

_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def main():
    dataset = CoffeeDataset("train", transform=_TRANSFORM)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=False)

    mean = torch.zeros(3)
    var  = torch.zeros(3)
    n    = 0

    for images, _ in loader:
        b = images.size(0)
        images = images.view(b, 3, -1)
        mean += images.mean(dim=(0, 2)) * b
        var  += images.var(dim=(0, 2), unbiased=False).mul(b)
        n += b

    mean /= n
    std = (var / n).sqrt()

    print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")


if __name__ == "__main__":
    main()
