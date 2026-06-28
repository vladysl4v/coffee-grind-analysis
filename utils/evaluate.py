"""
Evaluate a trained ConvNeXt-Small model on the real val and test sets.

Prints MAE/RMSE/bias/std/p80/p95, saves predictions CSV and charts for each split.

Usage:
    python evaluate.py path/to/best_model.pt
    python evaluate.py path/to/best_model.pt --crop 420
"""

import argparse
import csv
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


def _make_transform(crop: int):
    return transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class ValDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(_IMAGES_DIR / fname).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def _load_csv(path: Path) -> list[tuple[str, float]]:
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def evaluate(model_path: Path, split: str, device: torch.device, model: torch.nn.Module, crop: int = 420):
    samples = _load_csv(_LABELS_DIR / f"{split}.csv")
    loader  = DataLoader(ValDataset(samples, _make_transform(crop)),
                         batch_size=32, shuffle=False, num_workers=0)

    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds_all.append(model(imgs.to(device)).view(-1).cpu().numpy())
            gt_all.append(labels.numpy())

    preds  = np.concatenate(preds_all) * 100
    gt     = np.concatenate(gt_all)    * 100
    errors = preds - gt

    mae  = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()
    std  = errors.std()
    p80  = np.percentile(np.abs(errors), 80)
    p95  = np.percentile(np.abs(errors), 95)

    print(f"\n--- {split} ---")
    print(f"MAE:  {mae:.2f}%")
    print(f"RMSE: {rmse:.2f}%")
    print(f"Bias: {bias:+.2f}%")
    print(f"Std:  {std:.2f}%")
    print(f"p80:  {p80:.2f}%")
    print(f"p95:  {p95:.2f}%")

    out_dir = model_path.parent
    csv_out = out_dir / f"eval_predictions_{split}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted", "error"])
        w.writerows(zip(gt.tolist(), preds.tolist(), errors.tolist()))
    print(f"Predictions saved to {csv_out}")

    _plot_charts(gt, preds, errors, mae, rmse, bias, std, p80, p95, out_dir, split)

    return mae, rmse, bias, std, p80, p95


def _plot_charts(gt, preds, errors, mae, rmse, bias, std, p80, p95, out_dir: Path, split: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    lims = [min(gt.min(), preds.min()) - 2, max(gt.max(), preds.max()) + 2]
    ax.scatter(gt, preds, alpha=0.7, color="#4C72B0", edgecolors="white", linewidths=0.4, s=60)
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground truth (%)"); ax.set_ylabel("Predicted (%)")
    ax.set_title(f"{split} scatter — MAE={mae:.2f}%  RMSE={rmse:.2f}%")
    ax.legend(); fig.tight_layout()
    out = out_dir / f"eval_scatter_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(errors, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(bias, color="#44AA44", lw=1.2, ls="-",  label=f"Bias ({bias:+.2f}%)")
    ax.axvline( std, color="#DD4444", lw=1.2, ls="--", label=f"+1σ ({std:.2f}%)")
    ax.axvline(-std, color="#DD4444", lw=1.2, ls="--", label="-1σ")
    ax.axvline( p80, color="#FF9900", lw=1.0, ls=":",  label=f"p80 (±{p80:.2f}%)")
    ax.axvline(-p80, color="#FF9900", lw=1.0, ls=":")
    ax.axvline( p95, color="#CC6600", lw=1.0, ls=":",  label=f"p95 (±{p95:.2f}%)")
    ax.axvline(-p95, color="#CC6600", lw=1.0, ls=":")
    ax.set_xlabel("Error (predicted − ground truth, %)"); ax.set_ylabel("Count")
    ax.set_title(f"{split} error distribution"); ax.legend(fontsize=8); fig.tight_layout()
    out = out_dir / f"eval_error_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to best_model.pt")
    parser.add_argument("--crop", type=int, default=420)
    args = parser.parse_args()

    model_path = Path(args.model)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = get_model(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()

    evaluate(model_path, "val",  device, model, crop=args.crop)
    evaluate(model_path, "test", device, model, crop=args.crop)


if __name__ == "__main__":
    main()
