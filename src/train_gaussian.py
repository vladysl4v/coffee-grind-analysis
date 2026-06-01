"""
Train ConvNeXt-Small with Gaussian NLL loss for probabilistic fineness regression.

The model predicts mean (mu) and variance (sigma^2) per image.
Loss: Gaussian NLL = 0.5 * (log(var) + (y - mu)^2 / var)

MAE is tracked using mu, so it is directly comparable to standard regression runs.
Mean predicted sigma is logged each epoch to monitor uncertainty calibration.

Usage:
    uv run python src/train_gaussian.py --epochs 100 --unfreeze-after 15 --adamw --cosine-lr
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
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent))

from augmentation import ApplyAugmentation, seed_worker
from data_loader import _NORMALIZE, DEFAULT_EVAL_TRANSFORM, DEFAULT_TRAIN_TRANSFORM, get_loaders
from models.convnext_gaussian import ConvNeXtSmallGaussian

_ROOT     = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "data" / "runs" / "convnext_small_gaussian"


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian NLL training for ConvNeXt-Small")
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--batch-size",          type=int,   default=32)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--unfreeze",            action="store_true",
                   help="train fully unfrozen from epoch 1")
    p.add_argument("--unfreeze-after",      type=int,   default=None, metavar="N",
                   help="freeze N epochs then unfreeze backbone at lr/10")
    p.add_argument("--online-augment",      action="store_true")
    p.add_argument("--adamw",               action="store_true")
    p.add_argument("--cosine-lr",           action="store_true")
    p.add_argument("--early-stop-patience", type=int,   default=20)
    p.add_argument("--two-phase",           action="store_true",
                   help="phase 1: train mu head with MSE only; phase 2: full Gaussian NLL")
    p.add_argument("--explicit-sigma",      action="store_true",
                   help="supervise sigma directly with |mu - y| per sample instead of Gaussian NLL")
    p.add_argument("--phase1-epochs",       type=int,   default=15, metavar="N",
                   help="how many MSE-only epochs before switching to NLL (used with --two-phase)")
    return p.parse_args()


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
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_predictions(model, val_loader, device, epoch, run_dir):
    model.eval()
    ys, mus, sigmas = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with autocast(device_type=device.type):
                mu, log_var = model(images)
            sigma = F.softplus(log_var).sqrt()
            mus.append(mu.float().cpu() * 100)
            sigmas.append(sigma.float().cpu() * 100)
            ys.append(labels * 100)
    ys     = torch.cat(ys).numpy()
    mus    = torch.cat(mus).numpy()
    sigmas = torch.cat(sigmas).numpy()

    path = run_dir / f"predictions_epoch_{epoch:03d}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted_mu", "predicted_sigma"])
        for gt, mu, sigma in zip(ys, mus, sigmas):
            w.writerow([gt, mu, sigma])


def _save_scatter(model, val_loader, device, epoch, out_path):
    model.eval()
    ys, mus, sigmas = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with autocast(device_type=device.type):
                mu, log_var = model(images)
            sigma = F.softplus(log_var).sqrt()
            mus.append(mu.float().cpu() * 100)
            sigmas.append(sigma.float().cpu() * 100)
            ys.append(labels * 100)
    ys     = torch.cat(ys).numpy()
    mus    = torch.cat(mus).numpy()
    sigmas = torch.cat(sigmas).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Epoch {epoch}", fontsize=13)

    ax = axes[0]
    ax.errorbar(ys, mus, yerr=sigmas, fmt="o", alpha=0.6,
                elinewidth=0.8, capsize=2, markersize=4)
    lims = [min(ys.min(), mus.min()) - 2, max(ys.max(), mus.max()) + 2]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth (%)")
    ax.set_ylabel("Predicted μ (%)")
    ax.set_title("μ vs Truth  (error bars = ±σ)")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.hist(sigmas, bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Predicted σ (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Uncertainty distribution  (mean σ = {sigmas.mean():.2f}%)")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()

    if args.unfreeze_after is not None:
        args.unfreeze = False

    run_dir    = _next_run_dir()
    graphs_dir = run_dir / "graphs"
    models_dir = run_dir / "models"
    graphs_dir.mkdir()
    models_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model: convnext_small_gaussian | Device: {device} | Run: {run_dir.name}")

    train_transform = (
        T.Compose([ApplyAugmentation(), T.CenterCrop(224), T.ToTensor(), _NORMALIZE])
        if args.online_augment else DEFAULT_TRAIN_TRANSFORM
    )
    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        worker_init_fn=seed_worker if args.online_augment else None,
    )

    model = ConvNeXtSmallGaussian(freeze_backbone=not args.unfreeze).to(device)
    if args.two_phase:
        for param in model.log_var_head.parameters():
            param.requires_grad = False
    criterion_nll = nn.GaussianNLLLoss(eps=1e-6)
    criterion_mse = nn.MSELoss()
    criterion     = criterion_mse if args.two_phase else criterion_nll
    optimizer = (optim.AdamW if args.adamw else optim.Adam)(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.cosine_lr else
        optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    )
    scaler = GradScaler(device=device.type)

    config = {
        "model": "convnext_small_gaussian",
        "epochs": args.epochs, "lr": args.lr, "weight_decay": 1e-4,
        "batch_size": args.batch_size, "unfreeze": args.unfreeze,
        "unfreeze_after": args.unfreeze_after, "online_augment": args.online_augment,
        "loss": (
            "explicit_sigma(mse+residual)" if args.explicit_sigma
            else f"two_phase(mse→nll,phase1={args.phase1_epochs})" if args.two_phase
            else "gaussian_nll"
        ),
        "optimizer": "adamw" if args.adamw else "adam",
        "scheduler": "cosine" if args.cosine_lr else "plateau",
        "scheduler_patience": None if args.cosine_lr else 5,
        "early_stop_patience": args.early_stop_patience, "device": device.type,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_nll", "train_mae", "val_nll", "val_mae", "val_sigma_cal_err"]
        )

    train_nlls, val_nlls = [], []
    train_maes, val_maes = [], []
    best_val_mae    = float("inf")
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):

        # unfreeze backbone after N epochs
        if args.unfreeze_after is not None and epoch == args.unfreeze_after + 1:
            for param in model.parameters():
                param.requires_grad = True
            if args.two_phase:
                # keep log_var_head frozen — phase 2 block below owns it
                for param in model.log_var_head.parameters():
                    param.requires_grad = False
            head_ids     = {id(p) for p in optimizer.param_groups[0]["params"]}
            log_var_ids  = {id(p) for p in model.log_var_head.parameters()} if args.two_phase else set()
            backbone_params = [
                p for p in model.parameters()
                if id(p) not in head_ids and id(p) not in log_var_ids
            ]
            optimizer.add_param_group(
                {"params": backbone_params, "lr": args.lr / 10, "weight_decay": 1e-4}
            )
            scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch, eta_min=1e-6
                )
                if args.cosine_lr else
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )
            )
            print(f"Epoch {epoch}: backbone unfrozen, lr → {args.lr / 10:.2e}")

        # ── phase transition ─────────────────────────────────────────────────
        if args.two_phase and epoch == args.phase1_epochs + 1:
            for param in model.log_var_head.parameters():
                param.requires_grad = True
            optimizer.add_param_group(
                {"params": list(model.log_var_head.parameters()), "lr": args.lr, "weight_decay": 1e-4}
            )
            criterion = criterion_nll
            scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch, eta_min=1e-6
                )
                if args.cosine_lr else
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )
            )
            print(f"Epoch {epoch}: phase 2 — log_var_head unfrozen, switching loss to Gaussian NLL")

        # ── train ────────────────────────────────────────────────────────────
        model.train()
        acc_nll = acc_mae = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                mu, log_var = model(images)
                var = F.softplus(log_var)
                if args.explicit_sigma:
                    target_sigma = (mu.detach() - labels).abs()
                    loss = F.mse_loss(mu, labels) + F.mse_loss(var.sqrt(), target_sigma)
                elif criterion is criterion_mse:
                    loss = criterion_mse(mu, labels)
                else:
                    loss = criterion_nll(mu, labels, var)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_nll += loss.item()
                acc_mae += (mu.detach() - labels).abs().mean().item()
        train_nlls.append(acc_nll / len(train_loader))
        train_maes.append(acc_mae / len(train_loader))

        # ── val ──────────────────────────────────────────────────────────────
        model.eval()
        acc_nll = acc_mae = acc_sigma_err = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    mu, log_var = model(images)
                    var = F.softplus(log_var)
                    if args.explicit_sigma:
                        target_sigma = (mu.detach() - labels).abs()
                        acc_nll += (F.mse_loss(mu, labels) + F.mse_loss(var.sqrt(), target_sigma)).item()
                    elif criterion is criterion_mse:
                        acc_nll += criterion_mse(mu, labels).item()
                    else:
                        acc_nll += criterion_nll(mu, labels, var).item()
                actual_error = (mu.float() - labels).abs()
                acc_mae       += actual_error.mean().item()
                acc_sigma_err += (var.float().sqrt() - actual_error).abs().mean().item()
        val_nlls.append(acc_nll / len(val_loader))
        val_maes.append(acc_mae / len(val_loader))
        sigma_cal_err = acc_sigma_err / len(val_loader)

        scheduler.step() if args.cosine_lr else scheduler.step(val_maes[-1])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train nll={train_nlls[-1]:.4f} mae={train_maes[-1]*100:.2f} | "
            f"val nll={val_nlls[-1]:.4f} mae={val_maes[-1]*100:.2f} "
            f"sigma_err={sigma_cal_err*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_maes[-1] < best_val_mae:
            best_val_mae = val_maes[-1]
            early_stop_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            early_stop_counter += 1

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_nlls[-1], train_maes[-1],
                val_nlls[-1],   val_maes[-1],
                sigma_cal_err,
            ])

        _save_plot(train_nlls, val_nlls, "Gaussian NLL", graphs_dir / "nll.png")
        _save_plot(
            [x * 100 for x in train_maes],
            [x * 100 for x in val_maes],
            "MAE (%)", graphs_dir / "mae.png",
        )

        if epoch % 5 == 0:
            torch.save(model.state_dict(), models_dir / f"epoch_{epoch:03d}.pt")
            _save_predictions(model, val_loader, device, epoch, run_dir)
            _save_scatter(model, val_loader, device, epoch,
                          graphs_dir / f"scatter_{epoch:03d}.png")

        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best val MAE={best_val_mae*100:.2f}%)")
            break

    print(f"\nBest val MAE: {best_val_mae*100:.2f}%  →  {run_dir}")


if __name__ == "__main__":
    main()
