"""
Train DINOv2 (Meta) for coffee grind fineness regression.

Separate from train.py — experimental, does not touch the main pipeline.

Usage:
    uv run python src/train_dinov2.py --variant s
    uv run python src/train_dinov2.py --variant b --epochs 100 --unfreeze-after 10 --adamw --cosine-lr
"""

import argparse
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

from data_loader import DEFAULT_EVAL_TRANSFORM, DEFAULT_TRAIN_TRANSFORM, get_loaders
from models.dinov2 import get_dinov2_b, get_dinov2_s

_ROOT     = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "data" / "runs"

BUILDERS = {"s": get_dinov2_s, "b": get_dinov2_b}
LR_DEFAULTS = {"s": 1e-3, "b": 1e-3}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",          choices=["s", "b"], default="b",
                   help="DINOv2 variant: s=ViT-S/14 (384-d), b=ViT-B/14 (768-d)")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--lr",               type=float, default=None,
                   help="head LR (default: 1e-3)")
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--unfreeze",         action="store_true",
                   help="train fully unfrozen from epoch 1")
    p.add_argument("--unfreeze-after",   type=int,   default=None, metavar="N",
                   help="freeze N epochs then unfreeze backbone at lr/10")
    p.add_argument("--adamw",            action="store_true")
    p.add_argument("--cosine-lr",        action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=20)
    return p.parse_args()


def _next_run_dir(model_key: str) -> Path:
    runs = _RUNS_DIR / model_key
    runs.mkdir(parents=True, exist_ok=True)
    existing = sorted(runs.glob("run_*"))
    idx = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir = runs / f"run_{idx:03d}"
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
    args   = parse_args()
    lr     = args.lr or LR_DEFAULTS[args.variant]
    model_key = f"dinov2_{args.variant}"

    if args.unfreeze_after is not None:
        args.unfreeze = False

    run_dir    = _next_run_dir(model_key)
    graphs_dir = run_dir / "graphs"
    models_dir = run_dir / "models"
    graphs_dir.mkdir(); models_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model: {model_key} | Device: {device} | Run: {run_dir.name}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=DEFAULT_TRAIN_TRANSFORM,
    )

    model = BUILDERS[args.variant](freeze_backbone=not args.unfreeze).to(device)

    optimizer = (optim.AdamW if args.adamw else optim.Adam)(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.cosine_lr else
        optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    )
    criterion = nn.L1Loss()
    scaler    = GradScaler(device=device.type)

    config = {
        "model": model_key, "variant": args.variant,
        "epochs": args.epochs, "lr": lr, "weight_decay": 1e-4,
        "batch_size": args.batch_size, "unfreeze": args.unfreeze,
        "unfreeze_after": args.unfreeze_after,
        "optimizer": "adamw" if args.adamw else "adam",
        "scheduler": "cosine" if args.cosine_lr else "plateau",
        "early_stop_patience": args.early_stop_patience,
        "device": device.type,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mae", "val_mae"])

    train_maes, val_maes = [], []
    best_val_mae       = float("inf")
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):

        if args.unfreeze_after is not None and epoch == args.unfreeze_after + 1:
            for param in model.parameters():
                param.requires_grad = True
            head_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
            # drop head LR to backbone LR to avoid blowing up features
            optimizer.param_groups[0]["lr"] = lr / 10
            optimizer.add_param_group(
                {"params": backbone_params, "lr": lr / 10, "weight_decay": 1e-4}
            )
            scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch, eta_min=1e-6
                )
                if args.cosine_lr else
                optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            )
            print(f"Epoch {epoch}: backbone unfrozen, all lr → {lr/10:.2e}")

        # ── train ────────────────────────────────────────────────────────
        model.train()
        acc_mae = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(images)
                loss  = criterion(preds, labels)
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

        scheduler.step() if args.cosine_lr else scheduler.step(val_maes[-1])

        print(
            f"Epoch {epoch}/{args.epochs} | "
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

        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best val MAE={best_val_mae*100:.2f}%)")
            break

    print(f"\nBest val MAE: {best_val_mae*100:.2f}%  →  {run_dir}")


if __name__ == "__main__":
    main()
