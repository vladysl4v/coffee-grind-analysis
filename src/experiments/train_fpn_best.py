"""
Train FPN ConvNeXt-Small with best hyperparameters from Ray Tune trial 14.

Usage:
    uv run python src/train_fpn_best.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DEFAULT_TRAIN_TRANSFORM, get_loaders
from models.fpn_resnet import FPNResnet18
from train import _fgsm_perturb

_ROOT     = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "data" / "runs" / "fpn_convnext"

CONFIG = {
    "lr":            0.00018489727531581979,
    "weight_decay":  1.1223385887600535e-05,
    "batch_size":    16,
    "adv_epsilon":   0.08970084138617881,
    "adv_weight":    0.558859471643346,
    "huber_delta":   0.2115229235705976,
    "num_workers":   0,
    "epochs":        100,
    "early_stop_patience": 20,
}


def _next_run_dir() -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_RUNS_DIR.glob("run_*"))
    idx = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir = _RUNS_DIR / f"run_{idx:03d}"
    run_dir.mkdir()
    return run_dir


def _save_plot(train_vals, val_vals, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_vals, label="Train")
    ax.plot(val_vals,   label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    run_dir    = _next_run_dir()
    graphs_dir = run_dir / "graphs"
    models_dir = run_dir / "models"
    graphs_dir.mkdir(); models_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: fpn_convnext | Device: {device} | Run: {run_dir.name}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        train_transform=DEFAULT_TRAIN_TRANSFORM,
    )

    model = FPNResnet18().to(device)

    criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler(device=device.type)

    with open(run_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mae", "val_mae"])

    train_maes, val_maes   = [], []
    best_val_mae           = float("inf")
    early_stop_counter     = 0

    for epoch in range(1, CONFIG["epochs"] + 1):

        # ── train ────────────────────────────────────────────────────────
        model.train()
        acc_mae = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(images).view(-1)
                clean_loss = criterion(preds, labels)
            adv_images = _fgsm_perturb(images, labels, model, criterion, CONFIG["adv_epsilon"], device)
            with autocast(device_type=device.type):
                adv_preds = model(adv_images).view(-1)
                adv_loss  = criterion(adv_preds, labels)
            loss = (1 - CONFIG["adv_weight"]) * clean_loss + CONFIG["adv_weight"] * adv_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_mae += (preds.detach() - labels).abs().mean().item()
        train_maes.append(acc_mae / len(train_loader))

        # ── val ──────────────────────────────────────────────────────────
        model.eval()
        acc_mae = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    preds = model(images).view(-1)
                acc_mae += (preds.float() - labels).abs().mean().item()
        val_maes.append(acc_mae / len(val_loader))

        scheduler.step(val_maes[-1])

        print(
            f"Epoch {epoch}/{CONFIG['epochs']} | "
            f"train mae={train_maes[-1]*100:.2f}% | "
            f"val mae={val_maes[-1]*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_maes[-1] < best_val_mae:
            best_val_mae       = val_maes[-1]
            early_stop_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            early_stop_counter += 1

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_maes[-1], val_maes[-1]])

        _save_plot(
            [x * 100 for x in train_maes],
            [x * 100 for x in val_maes],
            "MAE (%)", graphs_dir / "mae.png",
        )

        if epoch % 5 == 0:
            torch.save(model.state_dict(), models_dir / f"epoch_{epoch:03d}.pt")

        if CONFIG["early_stop_patience"] > 0 and early_stop_counter >= CONFIG["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch} (best val MAE={best_val_mae*100:.2f}%)")
            break

    print(f"\nBest val MAE: {best_val_mae*100:.2f}%  →  {run_dir}")


if __name__ == "__main__":
    main()
