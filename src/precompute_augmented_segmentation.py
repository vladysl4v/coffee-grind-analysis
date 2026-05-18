"""
Augment train images and save to a target directory.

With ``--segment``, each raw image is segmented first, then augmented (geometry only
if ``--no-photometric``). This avoids masking failures on black rotation wedges from
augmenting full-frame raw photos.

Usage
-----
uv run python src/precompute_augmented_segmentation.py --segment --no-photometric
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from augmentation import AugmentationConfig, augment_image
from image_ops import is_degenerate_image
from region_extraction import extract_region

_ROOT = Path(__file__).parent.parent
_RAW_DIR = _ROOT / "data" / "images" / "raw"
_SEG_DIR = _ROOT / "data" / "images" / "segmentation"
_OUT_DIR = _ROOT / "data" / "images" / "augmented_segmentation"
_RAW_OUT_DIR = _ROOT / "data" / "images" / "augmented_raw"
_LABELS_DIR = _ROOT / "data" / "labels"

_MAX_AUG_RETRIES = 12


def _apply_segmentation(pil_image: Image.Image) -> Image.Image:
    arr = np.asarray(pil_image.convert("RGB"))
    segmented = extract_region(arr, cropping=False).astype(np.uint8)
    return Image.fromarray(segmented)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def _build_aug_config(no_photometric: bool) -> AugmentationConfig:
    if no_photometric:
        return AugmentationConfig.geometric_only()
    return AugmentationConfig()


def _augment_for_save(
    raw_img: Image.Image,
    *,
    config: AugmentationConfig,
    apply_segment: bool,
) -> Image.Image:
    """Build one augmented training image, with optional segmentation."""

    if apply_segment:
        base = _apply_segmentation(raw_img)
        return augment_image(base, config=config)
    return augment_image(raw_img, config=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate augmented training images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=3,
        metavar="N",
        help="number of augmented variants per train image (default: 3)",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="save augmented raw images to augmented_raw/ (no segmentation)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="segment before augment (recommended); stores masked RGB in augmented_segmentation/",
    )
    parser.add_argument(
        "--no-photometric",
        action="store_true",
        help="geometry-only augmentations (recommended with --segment)",
    )
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
        if not img_path.is_file():
            continue
        dest = out_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"Copied {copied} original images from {copy_src.name}/ → {out_dir.name}/")

    aug_config = _build_aug_config(args.no_photometric)
    if args.no_photometric:
        print("Augmentation mode: geometric_only (colour and texture preserved)")
    if apply_segment:
        print("Pipeline: segment(raw) → geometric augment (no remask on rotated wedges)")

    header, train_rows = _read_csv(_LABELS_DIR / "train.csv")
    augmented_rows: list[list[str]] = []
    skipped_black = 0
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
                if is_degenerate_image(Image.open(out_path).convert("RGB")):
                    out_path.unlink()
                else:
                    augmented_rows.append([aug_name, label])
                    continue

            saved = False
            for attempt in range(_MAX_AUG_RETRIES):
                try:
                    result = _augment_for_save(
                        raw_img,
                        config=aug_config,
                        apply_segment=apply_segment,
                    )
                    if is_degenerate_image(result):
                        skipped_black += 1
                        continue
                    result.save(out_path)
                    augmented_rows.append([aug_name, label])
                    saved = True
                    break
                except Exception as e:
                    print(f"[{i}/{total}] Failed {aug_name} (attempt {attempt + 1}): {e}")
            if not saved:
                print(f"[{i}/{total}] WARNING: could not save non-empty {aug_name}")

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed")

    _write_csv(out_csv, header, train_rows + augmented_rows)
    print(
        f"\nDone."
        f"\n  Augmented images : {len(augmented_rows)}"
        f"\n  Degenerate retries (discarded) : {skipped_black}"
        f"\n  Output CSV       : {out_csv.name} ({len(train_rows)} original + {len(augmented_rows)} augmented)"
        f"\n  Output dir       : {out_dir}"
    )


if __name__ == "__main__":
    main()
