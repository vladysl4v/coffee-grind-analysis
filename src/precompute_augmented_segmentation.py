"""
Augment train images and save to a target directory.

By default reads from raw/, applies all augmentations, and saves to augmented_segmentation/
alongside original segmented images.

Pass --raw-output to save augmented raw images to augmented_raw/ instead (no segmentation).
Pass --segment to apply background masking to each augmented image (augmented_segmentation/ only).
Pass --no-photometric to skip colour/lighting augmentations (recommended with --segment).

Usage
-----
uv run python src/precompute_augmented_segmentation.py
uv run python src/precompute_augmented_segmentation.py --augmentations-per-image 5
uv run python src/precompute_augmented_segmentation.py --segment --no-photometric
uv run python src/precompute_augmented_segmentation.py --raw-output
uv run python src/precompute_augmented_segmentation.py --raw-output --augmentations-per-image 5
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from augmentation import augment_image, AugmentationConfig
from region_extraction import extract_region

_ROOT       = Path(__file__).parent.parent
_RAW_DIR    = _ROOT / "data" / "images" / "raw"
_SEG_DIR    = _ROOT / "data" / "images" / "segmentation"
_OUT_DIR    = _ROOT / "data" / "images" / "augmented_segmentation"
_RAW_OUT_DIR = _ROOT / "data" / "images" / "augmented_raw"
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
    parser.add_argument("--raw-output", action="store_true",
                        help="save augmented raw images to augmented_raw/ (no segmentation); "
                             "use with --raw --augmented-data when training")
    parser.add_argument("--segment", action="store_true",
                        help="apply background segmentation to each augmented image (ignored with --raw-output)")
    parser.add_argument("--no-photometric", action="store_true",
                        help="disable colour/lighting augmentations (brightness, contrast, saturation, sharpness, blur, noise, jpeg); "
                             "recommended when using --segment to avoid breaking colour-based masking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = _RAW_OUT_DIR if args.raw_output else _OUT_DIR
    copy_src = _RAW_DIR if args.raw_output else _SEG_DIR
    out_csv = _LABELS_DIR / ("augmented_raw_train.csv" if args.raw_output else "augmented_train.csv")
    apply_segment = args.segment and not args.raw_output

    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in copy_src.iterdir():
        dest = out_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"Copied {copied} original images from {copy_src.name}/ → {out_dir.name}/")

    aug_config = AugmentationConfig(
        brightness_prob=0.0,
        contrast_prob=0.0,
        saturation_prob=0.0,
        sharpness_prob=0.0,
        blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
    ) if args.no_photometric else AugmentationConfig()

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
            out_path = out_dir / aug_name

            if out_path.exists():
                augmented_rows.append([aug_name, label])
                continue

            try:
                result = augment_image(raw_img, config=aug_config)
                if apply_segment:
                    result = _apply_segmentation(result)
                result.save(out_path)
                augmented_rows.append([aug_name, label])
            except Exception as e:
                print(f"[{i}/{total}] Failed {aug_name}: {e}")

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed")

    _write_csv(out_csv, header, train_rows + augmented_rows)
    print(
        f"\nDone."
        f"\n  Augmented images : {len(augmented_rows)}"
        f"\n  Output CSV       : {out_csv.name} ({len(train_rows)} original + {len(augmented_rows)} augmented)"
        f"\n  Output dir       : {out_dir}"
    )


if __name__ == "__main__":
    main()
