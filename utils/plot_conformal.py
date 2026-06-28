"""
Split conformal regression evaluation with charts.

Calibrates on val, evaluates on test. Produces:
  - conformal_errorbars.png      — per-sample intervals (green=covered, red=missed)
  - conformal_pred_vs_truth.png  — scatter coloured by coverage
  - conformal_coverage.png       — empirical vs nominal coverage across alpha values
  - conformal_width.png          — interval width vs nominal coverage across alpha values

Usage:
    python plot_conformal.py path/to/best_model.pt
    python plot_conformal.py path/to/best_model.pt --alpha 0.05  # 95% intervals
    python plot_conformal.py path/to/best_model.pt --crop 420
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import get_model

MEAN = [0.1557, 0.0899, 0.0404]
STD  = [0.0483, 0.0349, 0.0190]

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"


class SplitDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(_IMAGES_DIR / fname).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def _load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def _collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            ps.append(model(imgs.to(device)).view(-1).cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(ys) * 100, np.concatenate(ps) * 100


def _calibrate(y_true, y_pred, alpha):
    scores = np.abs(y_true - y_pred)
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to best_model.pt")
    parser.add_argument("--alpha", type=float, default=0.10, help="miscoverage rate (default 0.10 → 90% intervals)")
    parser.add_argument("--crop", type=int, default=420)
    args = parser.parse_args()

    model_path = Path(args.model)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)

    transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    lkw = dict(batch_size=32, shuffle=False, num_workers=0)
    y_val,  p_val  = _collect(model, DataLoader(SplitDataset(_load_csv(_LABELS_DIR / "val.csv"),  transform), **lkw), device)
    y_test, p_test = _collect(model, DataLoader(SplitDataset(_load_csv(_LABELS_DIR / "test.csv"), transform), **lkw), device)

    half_width = _calibrate(y_val, p_val, args.alpha)
    lo = p_test - half_width
    hi = p_test + half_width
    covered  = (y_test >= lo) & (y_test <= hi)
    coverage = covered.mean()

    print(f"Alpha: {args.alpha} → {1-args.alpha:.0%} nominal coverage")
    print(f"Empirical coverage: {coverage:.1%}")
    print(f"Mean interval width: {(hi - lo).mean():.2f}%")

    out_dir = model_path.parent / "conformal"
    out_dir.mkdir(exist_ok=True)

    # error bars
    n = min(30, len(y_test))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * n)))
    for i in range(n):
        col = "#2ca02c" if covered[i] else "#d62728"
        ax.plot([lo[i], hi[i]], [i, i], color=col, lw=2.4, solid_capstyle="round", alpha=0.9)
        ax.scatter(y_test[i], i, color="#1f4f1f" if covered[i] else "#7f0000", s=44, zorder=4, edgecolors="white", lw=0.6)
        ax.scatter(p_test[i], i, color="#1f77b4", s=28, zorder=3, edgecolors="white", lw=0.5)
    ax.set_xlabel("Fineness (%)"); ax.set_title(f"Conformal intervals — test set (first {n} samples)")
    fig.tight_layout(); fig.savefig(out_dir / "conformal_errorbars.png", dpi=150); plt.close(fig)

    # scatter
    lims = [min(y_test.min(), p_test.min()) - 2, max(y_test.max(), p_test.max()) + 2]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test[covered],  p_test[covered],  s=40, c="#2ca02c", alpha=0.8, edgecolors="white", lw=0.5, label="Covered")
    ax.scatter(y_test[~covered], p_test[~covered], s=52, c="#d62728", alpha=0.9, edgecolors="white", lw=0.5, label="Missed")
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth (%)"); ax.set_ylabel("Predicted (%)")
    ax.set_title("Prediction vs Truth — conformal coverage"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "conformal_pred_vs_truth.png", dpi=150); plt.close(fig)

    # sweep
    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    nom, emp, widths = [], [], []
    for a in alphas:
        hw = _calibrate(y_val, p_val, a)
        l_, h_ = p_test - hw, p_test + hw
        nom.append(1 - a)
        emp.append(float(((y_test >= l_) & (y_test <= h_)).mean()))
        widths.append((h_ - l_).mean())
    nom, emp, widths = map(np.array, (nom, emp, widths))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(nom, emp, "o-", color="tab:blue", lw=2, ms=7, label="Empirical coverage")
    ax.plot([0, 1], [0, 1], "--", color="0.45", lw=1.2, label="Perfect calibration")
    ax.axhline(1 - args.alpha, color="tab:green", ls=":", lw=2, label=f"Target {1-args.alpha:.0%}")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (test)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Empirical vs nominal coverage")
    fig.tight_layout(); fig.savefig(out_dir / "conformal_coverage.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(nom, widths, "s-", color="tab:red", lw=2, ms=7, label="Mean interval width")
    ax.axvline(1 - args.alpha, color="tab:green", ls=":", lw=2, label=f"α={args.alpha}")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Mean interval width (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Interval width vs nominal coverage")
    fig.tight_layout(); fig.savefig(out_dir / "conformal_width.png", dpi=150); plt.close(fig)

    print(f"Charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
