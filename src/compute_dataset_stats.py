"""
Compute per-channel mean and std over the training set (pre-normalization).

Uses the same mask-aware crop as ``data_loader`` (see ``image_ops.MaskBoundingBoxCrop``).
Writes results to ``data/dataset_stats.json`` for automatic loading at import time.

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
    var = torch.zeros(3)
    n = 0

    for images, _ in loader:
        b = images.size(0)
        flat = images.view(b, 3, -1)
        mean += flat.mean(dim=(0, 2)) * b
        var += flat.var(dim=(0, 2), unbiased=False).mul(b)
        n += b

    if n == 0:
        raise ValueError("Training loader produced zero batches — check CSV and image paths.")

    mean /= n
    std = (var / n).sqrt()
    return [float(mean[i]) for i in range(3)], [float(std[i]) for i in range(3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset mean/std for normalization.")
    parser.add_argument("--raw", action="store_true",
                        help="compute over raw (unsegmented) images")
    parser.add_argument("--augmented-data", action="store_true",
                        help="compute over augmented_segmentation/ and augmented_train.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print mean/std only; do not write data/dataset_stats.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = _pre_tensor_transform()

    if args.augmented_data:
        from data_loader import _AUG_CSV

        csv_map = _AUG_CSV
        images_dir = _AUG_IMAGES_DIR
        source = "augmented_segmentation"
    elif args.raw:
        from data_loader import _CSV

        csv_map = _CSV
        images_dir = _RAW_IMAGES_DIR
        source = "raw"
    else:
        csv_map = None
        images_dir = None
        source = "segmentation"

    dataset = CoffeeDataset("train", transform=transform, csv_map=csv_map, images_dir=images_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )

    mean, std = compute_channel_stats(loader)
    print(f"Source: {source} | n_images={len(dataset)}")
    print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    if not args.dry_run:
        _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _STATS_PATH.is_file():
            payload = json.loads(_STATS_PATH.read_text(encoding="utf-8"))
        else:
            payload = {"crop": "mask_bounding_box_224"}
        payload["crop"] = "mask_bounding_box_224"
        payload[source] = {
            "n_images": len(dataset),
            "mean": mean,
            "std": std,
        }
        _STATS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Updated {_STATS_PATH} (key={source!r})")
        print("Restart training to reload normalization constants.")


if __name__ == "__main__":
    main()
