import argparse
import csv
import random
from itertools import groupby
from pathlib import Path

from quality_filter import filter_images

_ROOT      = Path(__file__).parent.parent
LABELS_DIR = _ROOT / "data" / "labels"
INPUT_FILE = LABELS_DIR / "labels_train2.csv"

IMG_DIRS = {
    "raw":         _ROOT / "data" / "images" / "raw",
    "segmentation": _ROOT / "data" / "images" / "segmentation",
}

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test gets the remainder

SEED = 42


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def group_consecutive(rows: list[list[str]], max_group_size: int = 4) -> list[list[list[str]]]:
    """Group consecutive rows that share the same fineness value (column 1).

    A group is a consecutive block of rows with the same fineness value.
    Blocks longer than max_group_size are split into chunks of max_group_size,
    so two different grain samples with the same fineness are never merged.
    """
    groups = []
    for _, g in groupby(rows, key=lambda r: r[1]):
        block = list(g)
        for i in range(0, len(block), max_group_size):
            groups.append(block[i:i + max_group_size])
    return groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split CSVs with quality filtering.",
        epilog=(
            "Examples:\n"
            "  uv run python src/split_labels.py --source segmentation\n"
            "  uv run python src/split_labels.py --source raw\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--source", choices=IMG_DIRS.keys(), default="segmentation",
                        help="image source for quality filtering:\n"
                             "  segmentation — use background-masked images (recommended)\n"
                             "  raw          — use original images (default: segmentation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_dir = IMG_DIRS[args.source]
    bright_min = 0.0 if args.source == "segmentation" else 50.0

    print(f"Filtering using: {args.source}")

    header, rows = read_csv(INPUT_FILE)

    passing = set(filter_images([row[0] for row in rows], img_dir, bright_min=bright_min))
    rows = [row for row in rows if row[0] in passing]
    print(f"After quality filter: {len(rows)} images remain")

    groups = group_consecutive(rows)
    print(f"Total samples: {len(rows)}  |  grain groups: {len(groups)}")

    rng = random.Random(SEED)
    rng.shuffle(groups)

    n = len(groups)
    n_train = round(n * TRAIN_RATIO)
    n_val   = round(n * VAL_RATIO)

    train_groups = groups[:n_train]
    val_groups   = groups[n_train : n_train + n_val]
    test_groups  = groups[n_train + n_val :]

    def flatten(gs):
        return [row for g in gs for row in g]

    train = flatten(train_groups)
    val   = flatten(val_groups)
    test  = flatten(test_groups)

    write_csv(LABELS_DIR / "labels_train.csv", header, train)
    write_csv(LABELS_DIR / "labels_val.csv",   header, val)
    write_csv(LABELS_DIR / "labels_test.csv",  header, test)

    print(
        f"Groups → train {len(train_groups)} | val {len(val_groups)} | test {len(test_groups)}\n"
        f"Rows   → train {len(train)} | val {len(val)} | test {len(test)}"
    )


if __name__ == "__main__":
    main()
