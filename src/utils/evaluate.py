"""
Evaluate a saved ConvNeXt-Small model on the real val set.
Saves a scatter plot, error distribution chart, and predictions CSV.

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/evaluate.py run_025
    uv run python src/evaluate.py run_069 --crop 420
    uv run python src/evaluate.py run_025 run_043 run_045
    uv run python src/evaluate.py data/runs/convnext_small/run_025/best_model.pt
"""

import argparse, csv, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.convnext import get_convnext_small

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"


def _make_transform(crop):
    return transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1557, 0.0899, 0.0404],
                             std =[0.0483, 0.0349, 0.0190]),
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


def load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(row.iloc[0], float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def evaluate(model_path, crop=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_convnext_small(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()

    samples = load_csv(_LABELS_DIR / "val.csv")
    loader  = DataLoader(ValDataset(samples, _make_transform(crop)),
                         batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds_all.append(model(imgs.to(device)).view(-1).cpu().numpy())
            gt_all.append(labels.numpy())

    preds  = np.concatenate(preds_all) * 100
    gt     = np.concatenate(gt_all)    * 100
    errors = preds - gt
    mae    = np.abs(errors).mean()
    rmse   = np.sqrt((errors ** 2).mean())
    mse    = (errors ** 2).mean()
    std    = errors.std()
    bias   = errors.mean()
    p80    = np.percentile(np.abs(errors), 80)
    p95    = np.percentile(np.abs(errors), 95)

    out_dir = model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── predictions CSV ───────────────────────────────────────────────────────
    with open(out_dir / "eval_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted", "error"])
        w.writerows(zip(gt.tolist(), preds.tolist(), errors.tolist()))

    # ── scatter ───────────────────────────────────────────────────────────────
    lims = [min(gt.min(), preds.min()) - 2, max(gt.max(), preds.max()) + 2]
    fig, ax = plt.subplots()
    ax.scatter(gt, preds, alpha=0.7, color="#4C72B0",
               edgecolors="white", linewidths=0.4, s=60)
    ax.plot(lims, lims, "k--", lw=1, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set(xlabel="Ground Truth (%)", ylabel="Predicted (%)",
           title=f"GT vs Predicted — MAE={mae:.2f}%  RMSE={rmse:.2f}%")
    ax.legend(fontsize=9)
    ax.text(0.05, 0.95,
            f"MAE:  {mae:.2f}%\nRMSE: {rmse:.2f}%\nMSE:  {mse:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
            fontfamily="monospace")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── error distribution ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(0,    color="black",   lw=1.2, ls="--", label="zero")
    ax.axvline(bias, color="#44AA44", lw=1.2, ls="-",  label=f"mean bias ({bias:+.2f}%)")
    ax.axvline( std, color="#DD4444", lw=1.2, ls="--", label=f"+1σ ({std:.2f}%)")
    ax.axvline(-std, color="#DD4444", lw=1.2, ls="--", label=f"−1σ")
    ax.axvline( p80, color="#FF9900", lw=1.0, ls=":",  label=f"80th pct (±{p80:.2f}%)")
    ax.axvline(-p80, color="#FF9900", lw=1.0, ls=":")
    ax.axvline( p95, color="#CC6600", lw=1.0, ls=":",  label=f"95th pct (±{p95:.2f}%)")
    ax.axvline(-p95, color="#CC6600", lw=1.0, ls=":")
    ax.set(xlabel="Error  predicted − ground truth (%)", ylabel="Count",
           title=f"Error distribution — MAE={mae:.2f}%  std={std:.2f}%")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mae, rmse, bias, std, p80, p95


def resolve_path(arg):
    p = Path(arg)
    if p.suffix == ".pt":
        return p
    run_dir = _ROOT / "data" / "runs" / "convnext_small" / p.name
    if not run_dir.exists():
        run_dir = p
    for name in ("best_model.pt", "final_best.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No best_model.pt or final_best.pt found in {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["run_025"],
                        help="run names or .pt paths to evaluate")
    parser.add_argument("--crop", type=int, default=224,
                        help="centre-crop size used during training (default: 224)")
    args = parser.parse_args()

    print(f"Crop: {args.crop}px\n")
    print(f"{'Run':<30}  {'MAE':>7}  {'RMSE':>7}  {'Bias':>7}  {'Std':>7}  {'p80':>7}  {'p95':>7}")
    print("-" * 78)
    for arg in args.runs:
        path = resolve_path(arg)
        mae, rmse, bias, std, p80, p95 = evaluate(path, crop=args.crop)
        print(f"{path.parent.name:<30}  {mae:>6.2f}%  {rmse:>6.2f}%  "
              f"{bias:>+6.2f}%  {std:>6.2f}%  {p80:>6.2f}%  {p95:>6.2f}%")
        print(f"  → {path.parent / 'eval_error_distribution.png'}")


if __name__ == "__main__":
    main()
