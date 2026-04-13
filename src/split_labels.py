import csv
import random
from itertools import groupby
from pathlib import Path

from quality_filter import filter_images

LABELS_DIR = Path(__file__).parent.parent / "data" / "labels"
IMG_DIR    = Path(__file__).parent.parent / "data" / "images" / "raw"
INPUT_FILE = LABELS_DIR / "labels_train2.csv"

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


def group_consecutive(rows: list[list[str]]) -> list[list[list[str]]]:
    """Group consecutive rows that share the same fineness value (column 1)."""
    groups = []
    for _, g in groupby(rows, key=lambda r: r[1]):
        groups.append(list(g))
    return groups


def main() -> None:
    header, rows = read_csv(INPUT_FILE)

    passing = set(filter_images([row[0] for row in rows], IMG_DIR))
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
