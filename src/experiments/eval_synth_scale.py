"""
Evaluate a ConvNeXt-Small model on synthetic images to diagnose scale error.

For each synthetic image computes:
  value_error = actual - predicted     (positive = model under-predicts)
  scale_error = actual / predicted     (>1 = model under-predicts)

Plots histograms and prints mean/std for both.

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/eval_synth_scale.py              # uses run_025
    uv run python src/eval_synth_scale.py run_041
"""

import csv
import sys
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

_ROOT        = Path(__file__).parent.parent
_SYNTH_DIR   = _ROOT / "data" / "images" / "synthetic"
_SYNTH_CSV   = _ROOT / "data" / "labels" / "synthetic.csv"
_RUNS_DIR    = _ROOT / "data" / "runs" / "convnext_small"

TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1557, 0.0899, 0.0404],
                         std =[0.0483, 0.0349, 0.0190]),
])


class SynthDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(_SYNTH_DIR / fname).convert("RGB")
        return TRANSFORM(img), torch.tensor(label, dtype=torch.float32)


def load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def resolve_run(arg):
    p = Path(arg)
    if p.suffix == ".pt":
        return p
    run_dir = _RUNS_DIR / p.name
    best = run_dir / "best_model.pt"
    if not best.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    return best


def main():
    run_arg = sys.argv[1] if len(sys.argv) > 1 else "run_025"
    model_path = resolve_run(run_arg)
    print(f"Model : {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_convnext_small(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()

    samples = load_csv(_SYNTH_CSV)
    loader  = DataLoader(SynthDataset(samples), batch_size=32, shuffle=False,
                         num_workers=0, pin_memory=True)

    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds_all.append(model(imgs).view(-1).cpu().numpy())
            gt_all.append(labels.numpy())

    preds = np.concatenate(preds_all) * 100
    gt    = np.concatenate(gt_all)    * 100

    value_error = gt - preds
    # guard against near-zero predictions
    safe_preds  = np.where(np.abs(preds) < 1e-3, 1e-3, preds)
    scale_error = gt / safe_preds

    out_dir = model_path.parent

    # write per-image CSV
    fnames = [s[0] for s in samples]
    csv_path = out_dir / "synth_scale_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "actual", "predicted", "value_error", "scale_error"])
        for fname, a, p, ve, se in zip(fnames, gt.tolist(), preds.tolist(),
                                        value_error.tolist(), scale_error.tolist()):
            w.writerow([fname, f"{a:.4f}", f"{p:.4f}", f"{ve:.4f}", f"{se:.4f}"])

    print(f"\nSynthetic images : {len(gt)}")
    print(f"\nValue error  (actual - predicted)")
    print(f"  mean : {value_error.mean():.3f}%")
    print(f"  std  : {value_error.std():.3f}%")
    print(f"\nScale error  (actual / predicted)")
    print(f"  mean : {scale_error.mean():.4f}")
    print(f"  std  : {scale_error.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(value_error, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].axvline(value_error.mean(), color="red", linewidth=1.5, linestyle="-", label=f"mean={value_error.mean():.2f}%")
    axes[0].set_title("Value Error  (actual − predicted)")
    axes[0].set_xlabel("Error (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].text(0.97, 0.95, f"std={value_error.std():.2f}%",
                 transform=axes[0].transAxes, ha="right", va="top")

    axes[1].hist(scale_error, bins=40, color="saddlebrown", edgecolor="white", linewidth=0.5)
    axes[1].axvline(1, color="black", linewidth=1, linestyle="--")
    axes[1].axvline(scale_error.mean(), color="red", linewidth=1.5, linestyle="-", label=f"mean={scale_error.mean():.4f}")
    axes[1].set_title("Scale Error  (actual / predicted)")
    axes[1].set_xlabel("Ratio")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].text(0.97, 0.95, f"std={scale_error.std():.4f}",
                 transform=axes[1].transAxes, ha="right", va="top")

    fig.suptitle(f"Synthetic scale analysis — {model_path.parent.name}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "synth_scale_analysis.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")
    print(f"CSV  saved to {csv_path}")


if __name__ == "__main__":
    main()
