"""
Train ConvNeXt-Small for coffee grind fineness regression on raw images.

Replicates the hyperparameters of run_025 (segmented) but loads from the
raw image directory, using ImageNet normalisation appropriate for natural images.

Usage:
    uv run python src/train_convnext_small.py
    uv run python src/train_convnext_small.py --epochs 100 --unfreeze-after 15
    uv run python src/train_convnext_small.py --flip-augment
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import CoffeeDataset, _NORMALIZE_IMAGENET
from models.convnext import get_convnext_small

_ROOT     = Path(__file__).parent.parent
_RAW_DIR  = _ROOT / "data" / "images" / "raw"
_RUNS_DIR = _ROOT / "data" / "runs" / "convnext_small"

_EVAL_TRANSFORM = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    _NORMALIZE_IMAGENET,
])

_TRAIN_TRANSFORM = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    _NORMALIZE_IMAGENET,
])

_FLIP_TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.CenterCrop(224),
    T.ToTensor(),
    _NORMALIZE_IMAGENET,
])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--batch-size",          type=int,   default=32)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--unfreeze",            action="store_true")
    p.add_argument("--unfreeze-after",      type=int,   default=15, metavar="N")
    p.add_argument("--images",              choices=["raw", "segmented"], default="raw",
                   help="image source (default: raw)")
    p.add_argument("--flip-augment",        action="store_true",
                   help="apply random horizontal + vertical flips during training")
    p.add_argument("--early-stop-patience", type=int,   default=20)
    return p.parse_args()


def _next_run_dir() -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(_RUNS_DIR.glob("run_*"))
    idx = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir = _RUNS_DIR / f"run_{idx:03d}"
    run_dir.mkdir()
    return run_dir


def _get_loaders(batch_size: int, num_workers: int, train_transform, eval_transform, images_dir) -> tuple:
    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    train_loader = DataLoader(
        CoffeeDataset("train", transform=train_transform, images_dir=images_dir),
        batch_size=batch_size, shuffle=True, **_loader_kwargs,
    )
    val_loader = DataLoader(
        CoffeeDataset("val", transform=eval_transform, images_dir=images_dir),
        batch_size=batch_size, shuffle=False, **_loader_kwargs,
    )
    return train_loader, val_loader


def _save_plot(train_vals, val_vals, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_vals, label="Train")
    ax.plot(val_vals,   label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_scatter(model, val_loader, device, epoch, graphs_dir, run_dir):
    model.eval()
    preds_list, gt_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                preds_list.append(model(images).view(-1).float().cpu().numpy())
            gt_list.append(labels.numpy())
    preds = np.concatenate(preds_list) * 100
    gt    = np.concatenate(gt_list) * 100
    mse   = np.mean((preds - gt) ** 2)

    with open(run_dir / f"predictions_epoch_{epoch:03d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted"])
        w.writerows(zip(gt.tolist(), preds.tolist()))

    fig, ax = plt.subplots()
    ax.scatter(gt, preds, alpha=0.5)
    ax.set_xlabel("Ground Truth"); ax.set_ylabel("Prediction")
    ax.set_title(f"GT vs Pred (epoch {epoch})")
    ax.text(0.05, 0.95, f"MSE: {mse:.4f}", transform=ax.transAxes, va="top")
    fig.tight_layout(); fig.savefig(graphs_dir / f"scatter_epoch_{epoch:03d}.png", dpi=120)
    plt.close(fig)


def main():
    args = parse_args()

    if args.unfreeze_after is not None:
        args.unfreeze = False

    run_dir    = _next_run_dir()
    graphs_dir = run_dir / "graphs"
    models_dir = run_dir / "models"
    graphs_dir.mkdir(); models_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model: convnext_small | Images: {args.images} | Device: {device} | Run: {run_dir.name}")
    if args.flip_augment:
        print("Augmentation: random horizontal + vertical flips")

    train_transform, eval_transform = _make_transforms(args.images, args.flip_augment)
    train_loader, val_loader = _get_loaders(
        args.batch_size, args.num_workers,
        train_transform, eval_transform, _IMAGES_DIRS[args.images],
    )

    model = get_convnext_small(freeze_backbone=not args.unfreeze).to(device)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    scaler    = GradScaler(device=device.type)

    config = {
        "model": "convnext_small",
        "images": args.images,
        "normalisation": "imagenet" if args.images == "raw" else "dataset",
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "unfreeze": args.unfreeze,
        "unfreeze_after": args.unfreeze_after,
        "loss": "mse",
        "optimizer": "adam",
        "scheduler": "plateau",
        "scheduler_patience": 5,
        "flip_augment": args.flip_augment,
        "early_stop_patience": args.early_stop_patience,
        "device": device.type,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_mse", "val_mae"])

    train_mses, train_maes = [], []
    val_mses,   val_maes   = [], []
    best_val_mae           = float("inf")
    early_stop_counter     = 0

    for epoch in range(1, args.epochs + 1):

        if args.unfreeze_after is not None and epoch == args.unfreeze_after + 1:
            for param in model.parameters():
                param.requires_grad = True
            head_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
            optimizer.add_param_group(
                {"params": backbone_params, "lr": args.lr / 10, "weight_decay": 1e-4}
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            print(f"Epoch {epoch}: backbone unfrozen, backbone lr → {args.lr/10:.2e}")

        # ── train ────────────────────────────────────────────────────────
        model.train()
        acc_mse = torch.zeros(1, device=device)
        acc_mae = torch.zeros(1, device=device)
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(images).view(-1)
                loss  = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_mse += loss.detach()
                acc_mae += (preds.detach() - labels).abs().mean()
        train_mses.append((acc_mse / len(train_loader)).item())
        train_maes.append((acc_mae / len(train_loader)).item())

        # ── val ──────────────────────────────────────────────────────────
        model.eval()
        acc_mse = torch.zeros(1, device=device)
        acc_mae = torch.zeros(1, device=device)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    preds = model(images).view(-1)
                acc_mse += criterion(preds.float(), labels)
                acc_mae += (preds.float() - labels).abs().mean()
        val_mses.append((acc_mse / len(val_loader)).item())
        val_maes.append((acc_mae / len(val_loader)).item())

        scheduler.step(val_mses[-1])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train mse={train_mses[-1]*10000:.2f} mae={train_maes[-1]*100:.2f}% | "
            f"val mse={val_mses[-1]*10000:.2f} mae={val_maes[-1]*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_maes[-1] < best_val_mae:
            best_val_mae       = val_maes[-1]
            early_stop_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            early_stop_counter += 1

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_mses[-1], train_maes[-1], val_mses[-1], val_maes[-1]
            ])

        _save_plot(
            [x * 10000 for x in train_mses],
            [x * 10000 for x in val_mses],
            "MSE Loss (×10⁴)", graphs_dir / "loss.png",
        )
        _save_plot(
            [x * 100 for x in train_maes],
            [x * 100 for x in val_maes],
            "MAE (%)", graphs_dir / "mae.png",
        )

        if epoch % 5 == 0:
            torch.save(model.state_dict(), models_dir / f"epoch_{epoch:03d}.pt")
            _save_scatter(model, val_loader, device, epoch, graphs_dir, run_dir)

        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best val MAE={best_val_mae*100:.2f}%)")
            break

    print(f"\nBest val MAE: {best_val_mae*100:.2f}%  →  {run_dir}")


if __name__ == "__main__":
    main()
