"""
Train a model on the coffee grind fineness dataset.

Usage
-----
uv run python src/train.py --model resnet18
uv run python src/train.py --model simple_cnn --epochs 100 --lr 1e-4
uv run python src/train.py --model resnet18 --unfreeze
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_loaders, DEFAULT_TRAIN_TRANSFORM
from models.simple_cnn import SimpleCNN
from models.resnet import get_resnet18
from pathlib import Path


_ROOT = Path(__file__).parent.parent


MODELS = {
    "simple_cnn": lambda args: SimpleCNN(),
    "resnet18":   lambda args: get_resnet18(freeze_backbone=not args.unfreeze),
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
                        help="fine-tune the full ResNet18 backbone instead of just the head")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = _ROOT / "data" / "statistics" / "train" / args.model
    run_id = len(sorted(base_dir.glob("run_*"))) + 1
    run_dir = base_dir / f"run_{run_id:03d}"
    models_dir = run_dir / "models"
    graphs_dir = run_dir / "graphs"
    models_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model} | Device: {device} | Run: {run_dir.name}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size,
        train_transform=DEFAULT_TRAIN_TRANSFORM,
    )

    model = MODELS[args.model](args).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_mse, val_mse = [], []
    train_mae, val_mae = [], []

    for epoch in range(args.epochs):
        model.train()
        total_mse, total_mae = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item()
            total_mae += (preds - labels).abs().mean().item()
        train_mse.append(total_mse / len(train_loader))
        train_mae.append(total_mae / len(train_loader))

        model.eval()
        total_mse, total_mae = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                total_mse += criterion(preds, labels).item()
                total_mae += (preds - labels).abs().mean().item()
        val_mse.append(total_mse / len(val_loader))
        val_mae.append(total_mae / len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs} | train mse={train_mse[-1]:.4f} mae={train_mae[-1]:.4f} | val mse={val_mse[-1]:.4f} mae={val_mae[-1]:.4f}")

        _save_plot(train_mse, val_mse, "MSE Loss", graphs_dir / "mse.png")
        _save_plot(train_mae, val_mae, "MAE", graphs_dir / "mae.png")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), models_dir / f"epoch_{epoch:03d}.pt")
            _save_scatter(model, val_loader, device, epoch, graphs_dir)


def _save_plot(train_vals, val_vals, title, path):
    plt.figure()
    plt.plot(train_vals, label="train")
    plt.plot(val_vals, label="val")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_scatter(model, val_loader, device, epoch, graphs_dir):
    model.eval()
    preds_list, gt_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds_list.append(model(images.to(device)).cpu().numpy())
            gt_list.append(labels.numpy())

    preds = np.concatenate(preds_list)
    gt    = np.concatenate(gt_list)
    mse   = np.mean((preds - gt) ** 2)

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
