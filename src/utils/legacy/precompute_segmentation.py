"""
Run once to precompute segmentation for all images.

Usage
-----
uv run python src/precompute_segmentation.py
"""

import numpy as np
from pathlib import Path
from PIL import Image

from region_extraction import extract_region

_ROOT          = Path(__file__).parent.parent
_IMG_DIR       = _ROOT / "data" / "images" / "raw"
_SEGMENTED_DIR = _ROOT / "data" / "images" / "segmentation"
_CSV_PATH      = _ROOT / "data" / "labels" / "labels_train2.csv"


def main() -> None:
    import csv

    with open(_CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))[1:]
    image_names = [row[0] for row in rows]

    _SEGMENTED_DIR.mkdir(exist_ok=True)

    total = len(image_names)
    for i, name in enumerate(image_names, 1):
        out_path = _SEGMENTED_DIR / name
        if out_path.exists():
            continue
        try:
            img = np.asarray(Image.open(_IMG_DIR / name).convert("RGB"))
            segmented = extract_region(img, cropping=False).astype(np.uint8)
            Image.fromarray(segmented).save(out_path)
        except Exception as e:
            print(f"[{i}/{total}] Failed {name}: {e}")
            continue
        print(f"[{i}/{total}] {name}")

    print(f"Done. Segmented images saved to {_SEGMENTED_DIR}")


if __name__ == "__main__":
    main()
