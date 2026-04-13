import os
import shutil
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Data
_ROOT = Path(__file__).parent.parent
_IMG_DIR = _ROOT / "data" / "images"
_LABELS_PATH = _ROOT / "data" / "labels" / "labels_train2.csv"
_OUT_DIR = _ROOT / "data" / "analysis_output"

# Limits
SHARP_LIMIT = 70.0
BRIGHT_MIN = 50
BRIGHT_MAX = 220

def get_metrics(path):
    img = cv2.imread(str(path))
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # CenterCrop
    y, x = h // 4, w // 4
    crop = gray[y:y*3, x:x*3]

    sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
    brightness = np.mean(gray)

    return sharpness, brightness

def main():
    # Folders
    folders = {
        "good": _OUT_DIR / "good",
        "remove": _OUT_DIR / "to_remove"
    }
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(_LABELS_PATH, sep=';')
    except Exception as e:
        print(f"CSV error: {e}")
        return

    print(f"Processing {len(df)} images...")
    stats = []

    for _, row in df.iterrows():
        name = row['Sample']
        f_path = _IMG_DIR / name
        val = row['Fineness']

        if not f_path.exists():
            continue

        sharp, bright = get_metrics(f_path)

        # Filtration
        if sharp is None:
            status, reason = "remove", "corrupted"
        elif sharp < SHARP_LIMIT:
            status, reason = "remove", f"blur_{sharp:.1f}"
        elif not (BRIGHT_MIN < bright < BRIGHT_MAX):
            status, reason = "remove", f"light_{bright:.1f}"
        else:
            status, reason = "good", "ok"

        # Putting in folders
        shutil.copy2(f_path, folders[status] / name)

        stats.append({
            "Sample": name,
            "Status": status,
            "Reason": reason,
            "Sharp": sharp,
            "Bright": bright,
            "Fineness": val
        })

    # Save report
    pd.DataFrame(stats).to_csv(_OUT_DIR / "report.csv", index=False)
    print("Done. Report saved.")

if __name__ == "__main__":
    main()