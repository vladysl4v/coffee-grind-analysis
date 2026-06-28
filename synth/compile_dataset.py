"""
Compile synthetic dataset from synth/images/ into data/images/synthetic/.

Reads all grind_NNNNN.png + .json pairs from synth/images/,
copies the images to data/images/synthetic/, and writes data/labels/synthetic.csv.

Run this after generate.py before training:
    python synth/compile_dataset.py [--max N]
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np

_ROOT   = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent / 'images'
OUT_DIR = _ROOT / 'data' / 'images' / 'synthetic'
OUT_CSV = _ROOT / 'data' / 'labels' / 'synthetic.csv'


def main():
    parser = argparse.ArgumentParser(description='Compile synth dataset')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of images to include (default: all)')
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    candidates = []
    for json_path in sorted(SRC_DIR.glob('*.json')):
        img_path = json_path.with_suffix('.png')
        if not img_path.exists():
            continue
        meta = json.loads(json_path.read_text())
        fineness = meta['stats']['fineness_pct']
        candidates.append((img_path, fineness))

    selected = candidates if args.max is None else candidates[:args.max]
    print(f'Found {len(candidates)} images, selecting {len(selected)}')

    rows = []
    for img_path, fineness in selected:
        shutil.copy2(img_path, OUT_DIR / img_path.name)
        rows.append((img_path.name, fineness))

    with open(OUT_CSV, 'w') as f:
        f.write('Sample;Fineness\n')
        for name, fineness in rows:
            f.write(f'{name};{f"{fineness:.4f}".replace(".", ",")}\n')

    vals = np.array([r[1] for r in rows])
    print(f'Copied {len(rows)} images to {OUT_DIR}')
    print(f'Fineness range: {vals.min():.1f}% – {vals.max():.1f}%  Mean: {vals.mean():.1f}%')


if __name__ == '__main__':
    main()
