import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from numerical_features import FEATURE_NAMES, extract_numerical_features


_ROOT = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "segmentation"
_LABELS_DIR = _ROOT / "data" / "labels"
_FEATURES_DIR = _ROOT / "data" / "features" / "numerical"

_CSV = {
    "train": _LABELS_DIR / "train.csv",
    "val": _LABELS_DIR / "val.csv",
    "test": _LABELS_DIR / "test.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute numerical features for dataset splits and save them to CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python src/precompute_numerical_features.py\n"
            "  uv run python src/precompute_numerical_features.py --splits train val test\n"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=sorted(_CSV.keys()),
        help="dataset splits to precompute (default: train val)",
    )
    return parser.parse_args()


def _load_split_samples(split: str) -> list[str]:
    csv_path = _CSV[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run src/split_labels.py first.")

    df = pd.read_csv(csv_path, sep=";", decimal=",")
    return df.iloc[:, 0].tolist()


def _extract_row(sample: str) -> dict[str, float | str]:
    img_path = _IMAGES_DIR / sample
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    features = extract_numerical_features(image, return_dict=True)
    return {"sample": sample, **features}


def main():
    args = parse_args()
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        samples = _load_split_samples(split)
        print(f"Precomputing {split} features for {len(samples)} images", flush=True)

        rows = []
        for idx, sample in enumerate(samples, start=1):
            rows.append(_extract_row(sample))
            if idx == 1 or idx % 25 == 0 or idx == len(samples):
                print(f"{split}: {idx}/{len(samples)}", flush=True)

        df = pd.DataFrame(rows, columns=["sample", *FEATURE_NAMES])
        out_path = _FEATURES_DIR / f"{split}_features.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
