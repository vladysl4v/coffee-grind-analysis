"""
Split conformal regression evaluation with charts.

Calibrates on val, evaluates on test, produces 3 figures:
  - conformal_errorbars.png  — per-sample intervals (green=covered, red=missed)
  - conformal_pred_vs_truth.png — scatter coloured by coverage
  - conformal_coverage.png   — empirical vs nominal coverage across alpha values
  - conformal_width.png      — interval width vs nominal coverage across alpha values

Usage:
    uv run python src/plot_conformal.py --model convnext_small --run run_043
    uv run python src/plot_conformal.py --model convnext_small          # auto best run
    uv run python src/plot_conformal.py --model convnext_small --alpha 0.05  # 95% intervals
"""

import sys
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import CoffeeDataset, DEFAULT_EVAL_TRANSFORM

_ROOT     = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "data" / "runs"

MODEL_BUILDERS = {
    "convnext_small":  lambda: __import__("models.convnext",     fromlist=["get_convnext_small"]).get_convnext_small(freeze_backbone=False),
    "convnext_base":   lambda: __import__("models.convnext",     fromlist=["get_convnext_base"]).get_convnext_base(freeze_backbone=False),
    "resnet152":       lambda: __import__("models.resnet",       fromlist=["get_resnet152"]).get_resnet152(freeze_backbone=False),
    "resnet50":        lambda: __import__("models.resnet",       fromlist=["get_resnet50"]).get_resnet50(freeze_backbone=False),
    "resnet18":        lambda: __import__("models.resnet",       fromlist=["get_resnet18"]).get_resnet18(freeze_backbone=False),
    "efficientnet_b0": lambda: __import__("models.efficientnet", fromlist=["get_efficientnet_b0"]).get_efficientnet_b0(freeze_backbone=False),
    "vit":             lambda: __import__("models.vit",          fromlist=["get_vit"]).get_vit(freeze_backbone=False),
}


# ── conformal math ────────────────────────────────────────────────────────────

def calibrate(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    scores = np.abs(y_true - y_pred)
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def predict_interval(y_pred: np.ndarray, half_width: float):
    return y_pred - half_width, y_pred + half_width


# ── inference ─────────────────────────────────────────────────────────────────

def collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            ps.append(model(images).cpu().numpy().flatten())
            ys.append(labels.numpy().flatten())
    return np.concatenate(ys), np.concatenate(ps)


def find_best_run(model_dir: Path) -> Path:
    best_run, best_mae = None, float("inf")
    for run_dir in sorted(model_dir.iterdir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            rows = list(csv.DictReader(f))
        if not rows or "val_mae" not in rows[0]:
            continue
        mae = min(float(r["val_mae"]) for r in rows)
        if mae < best_mae:
            best_mae, best_run = mae, run_dir
    return best_run


# ── charts ────────────────────────────────────────────────────────────────────

def plot_errorbars(y_test, p_test, lo, hi, out_path, max_samples=30):
    y = y_test[:max_samples]
    p = p_test[:max_samples]
    l = lo[:max_samples]
    h = hi[:max_samples]
    covered = (y >= l) & (y <= h)
    n = len(y)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * n)))
    for i in range(n):
        col = "#2ca02c" if covered[i] else "#d62728"
        ax.plot([l[i], h[i]], [i, i], color=col, linewidth=2.4, solid_capstyle="round", alpha=0.9)
        ax.scatter(y[i], i, color="#1f4f1f" if covered[i] else "#7f0000", s=44, zorder=4, edgecolors="white", linewidths=0.6)
        ax.scatter(p[i], i, color="#1f77b4", s=28, zorder=3, edgecolors="white", linewidths=0.5)

    ax.scatter([], [], c="#1f4f1f", s=44, edgecolors="white", label="Truth — covered")
    ax.scatter([], [], c="#7f0000", s=44, edgecolors="white", label="Truth — missed")
    ax.scatter([], [], c="#1f77b4", s=28, edgecolors="white", label="Prediction")
    ax.plot([], [], color="#2ca02c", linewidth=2.4, label="Interval (covered)")
    ax.plot([], [], color="#d62728", linewidth=2.4, label="Interval (missed)")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i}" for i in range(n)])
    ax.set_xlabel("Fineness (%)")
    ax.set_title(f"Conformal intervals — test set (first {n} samples)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_truth(y_test, p_test, lo, hi, out_path):
    y = y_test
    p = p_test
    l = lo
    h = hi
    covered = (y >= l) & (y <= h)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y[covered],  p[covered],  s=40, c="#2ca02c", alpha=0.8, edgecolors="white", linewidths=0.5, label="Covered")
    ax.scatter(y[~covered], p[~covered], s=52, c="#d62728", alpha=0.9, edgecolors="white", linewidths=0.5, label="Missed")
    lims = [min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2]
    ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth (%)"); ax.set_ylabel("Predicted (%)")
    ax.set_title("Prediction vs Truth — conformal coverage")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coverage_width(sweep, alpha, out_path_coverage, out_path_width):
    nom    = np.array([r["nominal"]  for r in sweep])
    emp    = np.array([r["empirical"] for r in sweep])
    widths = np.array([r["width"]    for r in sweep])

    # Coverage chart
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(nom, emp, "o-", color="tab:blue", linewidth=2, markersize=7, label="Empirical coverage")
    ax.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.2, label="Perfect calibration")
    ax.axhline(1 - alpha, color="tab:green", linestyle=":", linewidth=2, label=f"Target {1-alpha:.0%} (α={alpha})")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (test)")
    ax.set_ylim(max(0.5, emp.min() - 0.05), 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_title("Empirical vs nominal coverage")
    fig.tight_layout()
    fig.savefig(out_path_coverage, dpi=150)
    plt.close(fig)

    # Width chart
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(nom, widths, "s-", color="tab:red", linewidth=2, markersize=7, label="Mean interval width")
    ax.axvline(1 - alpha, color="tab:green", linestyle=":", linewidth=2, label=f"α={alpha} ({1-alpha:.0%})")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean interval width (fineness %)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_title("Interval width vs nominal coverage")
    fig.tight_layout()
    fig.savefig(out_path_width, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_BUILDERS.keys())
    parser.add_argument("--run",   default=None)
    parser.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate (default 0.10 → 90% intervals)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_dir = _RUNS_DIR / args.model
    run_dir   = model_dir / args.run if args.run else find_best_run(model_dir)
    print(f"Run: {run_dir.name}")

    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        checkpoints = sorted((run_dir / "models").glob("*.pt")) if (run_dir / "models").exists() else []
        if not checkpoints:
            print(f"No checkpoint found in {run_dir}"); sys.exit(1)
        checkpoint = checkpoints[-1]
    print(f"Checkpoint: {checkpoint.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)

    def loader(split):
        ds = CoffeeDataset(split, transform=DEFAULT_EVAL_TRANSFORM)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_val,  p_val  = collect(model, loader("val"),  device)
    y_test, p_test = collect(model, loader("test"), device)

    # work in percentage space throughout
    y_val  *= 100; p_val  *= 100
    y_test *= 100; p_test *= 100

    half_width = calibrate(y_val, p_val, args.alpha)
    lo, hi     = predict_interval(p_test, half_width)

    covered  = (y_test >= lo) & (y_test <= hi)
    coverage = covered.mean()
    width    = (hi - lo).mean()

    print(f"\nAlpha:             {args.alpha} → {(1-args.alpha):.0%} nominal coverage")
    print(f"Empirical coverage: {coverage:.1%}")
    print(f"Mean interval width: {width:.2f}%")
    print(f"Calibration half-width: {half_width:.2f}%")

    # Alpha sweep
    sweep = []
    for a in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        hw = calibrate(y_val, p_val, a)
        l_, h_ = predict_interval(p_test, hw)
        cov = float(((y_test >= l_) & (y_test <= h_)).mean())
        sweep.append({"nominal": 1 - a, "empirical": cov, "width": (h_ - l_).mean()})

    out_dir = run_dir / "conformal"
    out_dir.mkdir(exist_ok=True)

    plot_errorbars(y_test, p_test, lo, hi, out_dir / "conformal_errorbars.png")
    plot_pred_vs_truth(y_test, p_test, lo, hi, out_dir / "conformal_pred_vs_truth.png")
    plot_coverage_width(sweep, args.alpha, out_dir / "conformal_coverage.png", out_dir / "conformal_width.png")

    print(f"\nCharts saved to {out_dir}/")


if __name__ == "__main__":
    main()
