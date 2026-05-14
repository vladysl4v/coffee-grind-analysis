"""
Run once to precompute segmentation for all images.

Reads RGB images from ``data/images/raw``, writes masked RGB to
``data/images/segmentation`` using ``region_extraction.extract_region``.

By default, if ``data/labels/labels_train2.csv`` exists, its ``Sample`` column
defines which files to process (legacy). Otherwise the union of filenames in
``train.csv``, ``val.csv``, and ``test.csv`` is used (typical workflow after
``split_labels.py``).

Usage
-----
uv run python src/precompute_segmentation.py
uv run python src/precompute_segmentation.py --source splits --overwrite
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from region_extraction import extract_region

_ROOT = Path(__file__).parent.parent
_IMG_DIR = _ROOT / "data" / "images" / "raw"
_SEGMENTED_DIR = _ROOT / "data" / "images" / "segmentation"
_LABELS_DIR = _ROOT / "data" / "labels"
_LABELS_TRAIN2 = _LABELS_DIR / "labels_train2.csv"


def _read_sample_column(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))[1:]
    return [row[0] for row in rows if row]


def collect_image_names(source: str) -> list[str]:
    """Return ordered unique filenames to segment."""
    if source == "labels_train2":
        if not _LABELS_TRAIN2.exists():
            raise FileNotFoundError(
                f"{_LABELS_TRAIN2} not found. Use --source splits or add labels_train2.csv."
            )
        names = _read_sample_column(_LABELS_TRAIN2)
        return list(dict.fromkeys(names))

    if source == "splits":
        seen: dict[str, None] = {}
        for split_name in ("train.csv", "val.csv", "test.csv"):
            path = _LABELS_DIR / split_name
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run src/split_labels.py first (or use --source labels_train2)."
                )
            for name in _read_sample_column(path):
                seen.setdefault(name, None)
        return list(seen.keys())

    if source == "auto":
        if _LABELS_TRAIN2.exists():
            return collect_image_names("labels_train2")
        return collect_image_names("splits")

    raise ValueError(f"Unknown --source {source!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute background-masked images.")
    parser.add_argument(
        "--source",
        choices=("auto", "splits", "labels_train2"),
        default="auto",
        help="which label CSV drives the file list (default: auto — labels_train2 if present, else splits)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-segment even when the output file already exists",
    )
    args = parser.parse_args()

    image_names = collect_image_names(args.source)
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    _SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    total = len(image_names)
    done = 0
    skipped = 0
    failed = 0

    for i, name in enumerate(image_names, 1):
        out_path = _SEGMENTED_DIR / name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        src = _IMG_DIR / name
        if not src.exists():
            print(f"[{i}/{total}] SKIP (missing raw): {name}")
            failed += 1
            continue
        try:
            img = np.asarray(Image.open(src).convert("RGB"))
            segmented = extract_region(img, cropping=False).astype(np.uint8)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(segmented).save(out_path)
            done += 1
            print(f"[{i}/{total}] {name}")
        except Exception as e:
            print(f"[{i}/{total}] Failed {name}: {e}")
            failed += 1

    print(
        f"Done. wrote={done} skipped_existing={skipped} missing_or_failed={failed} "
        f"out_dir={_SEGMENTED_DIR}"
    )


if __name__ == "__main__":
    main()
