"""
Augment raw train images and save to augmented_segmentation/.

By default augmented images are saved as-is (no segmentation applied).
Pass --segment to also apply background masking to each augmented image.

Copies original segmented val/test/train images into the same folder so the
data loader can point at a single directory regardless of split.

Usage
-----
uv run python src/precompute_augmented_segmentation.py
uv run python src/precompute_augmented_segmentation.py --augmentations-per-image 5
uv run python src/precompute_augmented_segmentation.py --segment
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from augmentation import augment_image
from region_extraction import extract_region

_ROOT       = Path(__file__).parent.parent
_RAW_DIR    = _ROOT / "data" / "images" / "raw"
_SEG_DIR    = _ROOT / "data" / "images" / "segmentation"
_OUT_DIR    = _ROOT / "data" / "images" / "augmented_segmentation"
_LABELS_DIR = _ROOT / "data" / "labels"


def _apply_segmentation(pil_image: Image.Image) -> Image.Image:
    arr = np.asarray(pil_image.convert("RGB"))
    segmented = extract_region(arr, cropping=False).astype(np.uint8)
    return Image.fromarray(segmented)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented training images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--augmentations-per-image", type=int, default=3, metavar="N",
                        help="number of augmented variants per train image (default: 3)")
    parser.add_argument("--segment", action="store_true",
                        help="apply background segmentation to each augmented image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # copy original segmented images (train + val + test) into output dir
    copied = 0
    for img_path in _SEG_DIR.iterdir():
        dest = _OUT_DIR / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"Copied {copied} original segmented images → {_OUT_DIR.name}/")

    header, train_rows = _read_csv(_LABELS_DIR / "train.csv")
    augmented_rows: list[list[str]] = []
    total = len(train_rows)

    for i, row in enumerate(train_rows, 1):
        name, label = row[0], row[1]
        raw_path = _RAW_DIR / name

        if not raw_path.exists():
            print(f"[{i}/{total}] Skipping {name} — not found in raw/")
            continue

        raw_img = Image.open(raw_path).convert("RGB")

        for aug_idx in range(args.augmentations_per_image):
            aug_name = f"{Path(name).stem}__aug_{aug_idx:03d}{Path(name).suffix}"
            out_path = _OUT_DIR / aug_name

            if out_path.exists():
                augmented_rows.append([aug_name, label])
                continue

            try:
                result = augment_image(raw_img)
                if args.segment:
                    result = _apply_segmentation(result)
                result.save(out_path)
                augmented_rows.append([aug_name, label])
            except Exception as e:
                print(f"[{i}/{total}] Failed {aug_name}: {e}")

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed")

    _write_csv(_LABELS_DIR / "augmented_train.csv", header, train_rows + augmented_rows)
    print(
        f"\nDone. {'Segmentation applied.' if args.segment else 'No segmentation (use --segment to enable).'}"
        f"\n  Augmented images    : {len(augmented_rows)}"
        f"\n  augmented_train.csv : {len(train_rows)} original + {len(augmented_rows)} augmented = {len(train_rows) + len(augmented_rows)} total"
        f"\n  Output dir          : {_OUT_DIR}"
    )


if __name__ == "__main__":
    main()
