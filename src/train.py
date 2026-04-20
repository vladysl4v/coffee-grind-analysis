"""
Train a model on the coffee grind fineness dataset.

Usage
-----
uv run python src/train.py --model resnet18
uv run python src/train.py --model simple_cnn --epochs 100 --lr 1e-4
uv run python src/train.py --model resnet18 --unfreeze
"""

import argparse
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_loaders, DEFAULT_TRAIN_TRANSFORM
from models.simple_cnn import SimpleCNN
from models.resnet import get_resnet152, get_resnet18
from models.efficientnet import get_efficientnet_b0
from models.vit import get_vit
from models.vit_large import get_vit_large
from models.resnext import get_resnext50
from pathlib import Path


_ROOT = Path(__file__).parent.parent


MODELS = {
    "simple_cnn": lambda args: SimpleCNN(),
    "resnet152":  lambda args: get_resnet152(freeze_backbone=not args.unfreeze),
    "resnet18":   lambda args: get_resnet18(freeze_backbone=not args.unfreeze),
    "vit":        lambda args: get_vit(freeze_backbone=not args.unfreeze),
    "vit_large":  lambda args: get_vit_large(freeze_backbone=not args.unfreeze),
    "resnext50":  lambda args: get_resnext50(freeze_backbone=not args.unfreeze),
    "efficientnet_b0": lambda args: get_efficientnet_b0(freeze_backbone=not args.unfreeze),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on the coffee grind fineness dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python src/train.py --model resnet18\n"
            "  uv run python src/train.py --model resnet18 --unfreeze --epochs 100\n"
            "  uv run python src/train.py --model simple_cnn --lr 1e-4 --batch-size 16\n"
        ),
    )
    parser.add_argument("--model", required=True, choices=MODELS.keys(),
                        help="model architecture to train:\n"
                             "  resnet18   — pretrained ResNet18, only the regression head is trained by default\n"
                             "  simple_cnn — lightweight 3-layer CNN trained from scratch")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of images per batch (default: 32)")
    parser.add_argument("--unfreeze", action="store_true",
                        help="train with backbone fully unfrozen from epoch 1")
    parser.add_argument("--unfreeze-after", type=int, default=None, metavar="N",
                        help="unfreeze backbone after N epochs and fine-tune at lr/10 (overrides --unfreeze)")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = _ROOT / "data" / "runs" / args.model
    run_id = len(sorted(base_dir.glob("run_*"))) + 1
    run_dir = base_dir / f"run_{run_id:03d}"
    models_dir = run_dir / "models"
    graphs_dir = run_dir / "graphs"
    models_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model: {args.model} | Device: {device} | Run: {run_dir.name}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size,
        train_transform=DEFAULT_TRAIN_TRANSFORM,
        num_workers=8
    )

    if args.unfreeze_after is not None:
        args.unfreeze = False
    model = MODELS[args.model](args).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(device=device.type)

    train_mse, val_mse = [], []
    train_mae, val_mae = [], []

    config = {
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "unfreeze": args.unfreeze,
        "unfreeze_after": args.unfreeze_after,
        "device": device.type,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_mse", "val_mae"])

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_after is not None and epoch == args.unfreeze_after + 1:
            for param in model.parameters():
                param.requires_grad = True
            backbone_params = [p for p in model.parameters() if not any(p is hp for hp in optimizer.param_groups[0]["params"])]
            optimizer.add_param_group({"params": backbone_params, "lr": args.lr / 10, "weight_decay": 1e-4})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            print(f"Epoch {epoch}: backbone unfrozen, lr → {args.lr / 10:.2e}")

        model.train()
        acc_mse = torch.zeros(1, device=device)
        acc_mae = torch.zeros(1, device=device)
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(images).view(-1)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_mse += loss.detach()
                acc_mae += (preds.detach() - labels).abs().mean()
        train_mse.append((acc_mse / len(train_loader)).item())
        train_mae.append((acc_mae / len(train_loader)).item())

        model.eval()
        acc_mse = torch.zeros(1, device=device)
        acc_mae = torch.zeros(1, device=device)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    preds = model(images).view(-1)
                    acc_mse += criterion(preds, labels)
                acc_mae += (preds - labels).abs().mean()
        val_mse.append((acc_mse / len(val_loader)).item())
        val_mae.append((acc_mae / len(val_loader)).item())

        scheduler.step(val_mse[-1])
        print(f"Epoch {epoch}/{args.epochs} | train mse={train_mse[-1]*10000:.2f} mae={train_mae[-1]*100:.2f} | val mse={val_mse[-1]*10000:.2f} mae={val_mae[-1]*100:.2f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_mse[-1], train_mae[-1], val_mse[-1], val_mae[-1]])

        _save_plot([x * 10000 for x in train_mse], [x * 10000 for x in val_mse], "MSE Loss", graphs_dir / "mse.png")
        _save_plot([x * 100 for x in train_mae], [x * 100 for x in val_mae], "MAE", graphs_dir / "mae.png")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), models_dir / f"epoch_{epoch:03d}.pt")
            _save_scatter(model, val_loader, device, epoch, graphs_dir, run_dir)


def _save_plot(train_vals, val_vals, title, path):
    plt.figure()
    plt.plot(train_vals, label="train")
    plt.plot(val_vals, label="val")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_scatter(model, val_loader, device, epoch, graphs_dir, run_dir):
    model.eval()
    preds_list, gt_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds_list.append(model(images.to(device)).view(-1).cpu().numpy())
            gt_list.append(labels.numpy())

    preds = np.concatenate(preds_list) * 100
    gt    = np.concatenate(gt_list) * 100
    mse   = np.mean((preds - gt) ** 2)

    with open(run_dir / f"predictions_epoch_{epoch:03d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted"])
        w.writerows(zip(gt.tolist(), preds.tolist()))

    plt.figure()
    plt.scatter(gt, preds, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(f"GT vs Pred (epoch {epoch})")
    plt.text(0.05, 0.95, f"MSE: {mse:.4f}", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(graphs_dir / f"scatter_epoch_{epoch:03d}.png")
    plt.close()


if __name__ == "__main__":
    main()
