import cv2
import numpy as np
from pathlib import Path


def filter_images(
    image_names: list[str],
    img_dir: Path,
    sharp_limit: float = 70.0,
    bright_min: float = 50.0,
    bright_max: float = 220.0,
) -> list[str]:
    """Return the subset of image_names that pass sharpness and brightness checks.

    Parameters
    ----------
    image_names: filenames to evaluate (as they appear in the CSV)
    img_dir:     directory where the images live
    sharp_limit: minimum Laplacian variance — images below this are considered blurry
    bright_min:  minimum mean brightness (0-255) — images below are too dark
    bright_max:  maximum mean brightness (0-255) — images above are overexposed
    """
    passing = []

    for name in image_names:
        path = img_dir / name
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        y, x = h // 4, w // 4
        crop = gray[y:y*3, x:x*3]

        sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
        brightness = np.mean(gray)

        if sharpness < sharp_limit:
            continue
        if not (bright_min < brightness < bright_max):
            continue

        passing.append(name)

    return passing


if __name__ == "__main__":
    import csv
    import shutil

    _ROOT         = Path(__file__).parent.parent
    _IMG_DIR      = _ROOT / "data" / "images" / "raw"
    _REJECTED_DIR = _ROOT / "data" / "images" / "rejected"
    _CSV_PATH     = _ROOT / "data" / "labels" / "labels_train2.csv"

    with open(_CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    header, rows = rows[0], rows[1:]
    all_names = [row[0] for row in rows]

    total = len(all_names)
    counts = {"corrupted": 0, "blurry": 0, "bad_brightness": 0, "ok": 0}

    for name in all_names:
        img = cv2.imread(str(_IMG_DIR / name))
        if img is None:
            counts["corrupted"] += 1
            shutil.copy2(_IMG_DIR / name, _REJECTED_DIR / name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        y, x = h // 4, w // 4
        crop = gray[y:y*3, x:x*3]

        sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
        brightness = np.mean(gray)

        if sharpness < 70.0:
            counts["blurry"] += 1
            shutil.copy2(_IMG_DIR / name, _REJECTED_DIR / name)
        elif not (50.0 < brightness < 220.0):
            counts["bad_brightness"] += 1
            shutil.copy2(_IMG_DIR / name, _REJECTED_DIR / name)
        else:
            counts["ok"] += 1

    print(f"Total images  : {total}")
    print(f"Corrupted     : {counts['corrupted']}")
    print(f"Blurry        : {counts['blurry']}")
    print(f"Bad brightness: {counts['bad_brightness']}")
    print(f"Passing       : {counts['ok']}  ({counts['ok']/total*100:.1f}%)")
