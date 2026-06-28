"""
Runs a model checkpoint on the validation set and plots error distribution.

Usage:
    uv run python src/plot_error_distribution.py --model convnext_small --run run_043
    uv run python src/plot_error_distribution.py --model convnext_small  # uses best run automatically
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import CoffeeDataset, DEFAULT_EVAL_TRANSFORM

_ROOT     = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "data" / "runs"

MODEL_BUILDERS = {
    "convnext_small":   lambda: __import__("models.convnext",    fromlist=["get_convnext_small"]).get_convnext_small(freeze_backbone=False),
    "convnext_base":    lambda: __import__("models.convnext",    fromlist=["get_convnext_base"]).get_convnext_base(freeze_backbone=False),
    "resnet152":        lambda: __import__("models.resnet",      fromlist=["get_resnet152"]).get_resnet152(freeze_backbone=False),
    "resnet50":         lambda: __import__("models.resnet",      fromlist=["get_resnet50"]).get_resnet50(freeze_backbone=False),
    "resnet18":         lambda: __import__("models.resnet",      fromlist=["get_resnet18"]).get_resnet18(freeze_backbone=False),
    "efficientnet_b0":  lambda: __import__("models.efficientnet",fromlist=["get_efficientnet_b0"]).get_efficientnet_b0(freeze_backbone=False),
    "vit":              lambda: __import__("models.vit",         fromlist=["get_vit"]).get_vit(freeze_backbone=False),
}


def find_best_run(model_dir: Path) -> Path:
    import csv
    best_run, best_mae = None, float("inf")
    for run_dir in sorted(model_dir.iterdir()):
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        with open(metrics) as f:
            rows = list(csv.DictReader(f))
        if not rows or "val_mae" not in rows[0]:
            continue
        mae = min(float(r["val_mae"]) for r in rows)
        if mae < best_mae:
            best_mae, best_run = mae, run_dir
    return best_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_BUILDERS.keys())
    parser.add_argument("--run",   default=None, help="e.g. run_043 — omit to use best run automatically")
    parser.add_argument("--crop",  type=int, default=224, help="centre-crop used during training (default: 224)")
    args = parser.parse_args()

    model_dir = _RUNS_DIR / args.model

    if args.run:
        run_dir = model_dir / args.run
    else:
        run_dir = find_best_run(model_dir)
        print(f"Best run: {run_dir.name}")

    for candidate in ["best_model.pt", "final_best.pt"]:
        checkpoint = run_dir / candidate
        if checkpoint.exists():
            break
    else:
        checkpoints = sorted((run_dir / "models").glob("*.pt")) if (run_dir / "models").exists() else []
        if not checkpoints:
            print(f"No checkpoint found in {run_dir}")
            sys.exit(1)
        checkpoint = checkpoints[-1]
        print(f"No best_model.pt/final_best.pt — using {checkpoint.name}")
    print(f"Checkpoint: {checkpoint.name}")

    out = run_dir / "graphs" / "error_distribution.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    from torchvision import transforms as T
    from data_loader import _NORMALIZE
    eval_transform = (DEFAULT_EVAL_TRANSFORM if args.crop == 224 else
                      T.Compose([T.CenterCrop(args.crop), T.ToTensor(), _NORMALIZE]))
    val_dataset = CoffeeDataset("val", transform=eval_transform)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    gts, preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out_batch = model(images).cpu().numpy().flatten() * 100
            gt_batch  = labels.numpy().flatten() * 100
            preds.extend(out_batch.tolist())
            gts.extend(gt_batch.tolist())

    errors   = np.array(preds) - np.array(gts)
    mae      = np.abs(errors).mean()
    rmse     = np.sqrt((errors ** 2).mean())
    abs_err  = np.abs(errors)
    p80      = np.percentile(abs_err, 80)
    p95      = np.percentile(abs_err, 95)
    std      = errors.std()
    mean_err = errors.mean()

    scatter_text = f"MAE:  {mae:.2f}%\nRMSE: {rmse:.2f}%"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{args.model} — {run_dir.name} (Val)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(errors, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(mean_err, color="#44AA44", linewidth=1.2, linestyle="-",  label=f"Mean error ({mean_err:+.2f})")
    ax.axvline( std,     color="#DD4444", linewidth=1.2, linestyle="--", label=f"+1σ ({std:.2f})")
    ax.axvline(-std,     color="#DD4444", linewidth=1.2, linestyle="--", label=f"−1σ")
    ax.axvline( p80,     color="#FF9900", linewidth=1.0, linestyle=":",  label=f"80th pct (±{p80:.2f})")
    ax.axvline(-p80,     color="#FF9900", linewidth=1.0, linestyle=":")
    ax.axvline( p95,     color="#CC6600", linewidth=1.0, linestyle=":",  label=f"95th pct (±{p95:.2f})")
    ax.axvline(-p95,     color="#CC6600", linewidth=1.0, linestyle=":")
    ax.axvline(errors.max(), color="#880000", linewidth=1.0, linestyle="-", label=f"Max ({errors.max():+.2f})")
    ax.axvline(errors.min(), color="#880000", linewidth=1.0, linestyle="-", label=f"Min ({errors.min():+.2f})")
    ax.set_xlabel("Error (predicted − ground truth, %)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    lims = [min(gts + preds) - 2, max(gts + preds) + 2]
    ax.scatter(gts, preds, alpha=0.7, color="#4C72B0", edgecolors="white", linewidths=0.4, s=60)
    ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth (%)")
    ax.set_ylabel("Predicted (%)")
    ax.set_title("Ground Truth vs Predicted")
    ax.legend()
    ax.text(
        0.05, 0.95, scatter_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        fontfamily="monospace",
    )

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    print(f"\nVal MAE:           {mae:.4f}")
    print(f"Mean error (bias): {mean_err:+.4f}")
    print(f"Val RMSE:          {rmse:.4f}")
    print(f"Std of errors:     {std:.4f}")
    print(f"80% within:       ±{p80:.2f}")
    print(f"95% within:       ±{p95:.2f}")
    print(f"Max overestimate: +{errors.max():.2f}")
    print(f"Max underestimate: {errors.min():.2f}")


if __name__ == "__main__":
    main()
