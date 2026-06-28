"""
Train ConvNeXt-Small on coffee grind fineness — best known configuration.

  - 420 px centre crop
  - Frozen backbone for first 15 epochs, then unfrozen at lr/10
  - Mixed real + synthetic training throughout
  - Early stopping (patience 12)

Expected data layout:
    data/images/raw/          ← real images
    data/images/synthetic/    ← synthetic images
    data/labels/train.csv     ← Sample;Fineness (semicolon, comma decimal)
    data/labels/val.csv
    data/labels/synthetic.csv

Saves best_model.pt to data/runs/run_NNN/.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from model import get_model

# ── Config ────────────────────────────────────────────────────────────────────

CROP             = 420
BATCH            = 32
LR               = 1e-3
EPOCHS           = 100
UNFREEZE_AFTER   = 15
EARLY_STOP       = 12
SYNTH_LIMIT      = 1500
SYNTH_VAL_FRAC   = 0.1
NUM_WORKERS      = 4

MEAN = [0.1557, 0.0899, 0.0404]
STD  = [0.0483, 0.0349, 0.0190]

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT             = Path(__file__).parent.parent
_REAL_IMAGES_DIR  = _ROOT / "data" / "images" / "raw"
_SYNTH_IMAGES_DIR = _ROOT / "data" / "images" / "synthetic"
_LABELS_DIR       = _ROOT / "data" / "labels"
_RUNS_DIR         = _ROOT / "data" / "runs"

# ── Transforms ────────────────────────────────────────────────────────────────

_normalize  = transforms.Normalize(mean=MEAN, std=STD)
_transform  = transforms.Compose([transforms.CenterCrop(CROP), transforms.ToTensor(), _normalize])

# ── Datasets ──────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[tuple[str, float]]:
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


class ImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, float]], images_dir: Path):
        self.samples   = samples
        self.images_dir = images_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        return _transform(img), torch.tensor(label, dtype=torch.float32)


def _build_loaders():
    real_train = _load_csv(_LABELS_DIR / "train.csv")
    real_val   = _load_csv(_LABELS_DIR / "val.csv")

    synth_all  = _load_csv(_LABELS_DIR / "synthetic.csv")
    rng        = np.random.default_rng(42)
    idx        = rng.permutation(len(synth_all))
    n_val      = max(1, int(len(synth_all) * SYNTH_VAL_FRAC))
    synth_val  = [synth_all[i] for i in idx[:n_val]]
    synth_train = [synth_all[i] for i in idx[n_val:]][:SYNTH_LIMIT]

    lkw = dict(num_workers=NUM_WORKERS, pin_memory=True,
               persistent_workers=NUM_WORKERS > 0,
               prefetch_factor=4 if NUM_WORKERS > 0 else None,
               multiprocessing_context="spawn" if NUM_WORKERS > 0 else None)

    real_train_ds  = ImageDataset(real_train,  _REAL_IMAGES_DIR)
    synth_train_ds = ImageDataset(synth_train, _SYNTH_IMAGES_DIR)
    mixed_ds       = ConcatDataset([real_train_ds, synth_train_ds])

    train_loader      = DataLoader(mixed_ds,                              batch_size=BATCH, shuffle=True,  **lkw)
    val_loader        = DataLoader(ImageDataset(real_val, _REAL_IMAGES_DIR), batch_size=BATCH, shuffle=False, **lkw)
    synth_val_loader  = DataLoader(ImageDataset(synth_val, _SYNTH_IMAGES_DIR), batch_size=BATCH, shuffle=False, **lkw)

    print(f"Real train: {len(real_train)}  Synth train: {len(synth_train)}  Real val: {len(real_val)}  Synth val: {len(synth_val)}")
    return train_loader, val_loader, synth_val_loader

# ── Training ──────────────────────────────────────────────────────────────────

def main():
    existing = sorted(_RUNS_DIR.glob("run_*"))
    run_id   = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir  = _RUNS_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, val_loader, synth_val_loader = _build_loaders()

    model     = get_model(freeze_backbone=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler    = GradScaler(device=device.type)

    best_val_mae     = float("inf")
    early_stop_count = 0

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_real_mse", "val_real_mae", "val_synth_mae"])

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_AFTER + 1:
            for param in model.parameters():
                param.requires_grad = True
            backbone_params = [p for p in model.parameters()
                               if not any(p is hp for hp in optimizer.param_groups[0]["params"])]
            optimizer.add_param_group({"params": backbone_params, "lr": LR / 10, "weight_decay": 1e-4})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            print(f"Epoch {epoch}: backbone unfrozen, lr → {LR/10:.2e}")

        # train
        model.train()
        acc_mse = acc_mae = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(imgs).view(-1)
                loss  = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_mse += loss.item()
                acc_mae += (preds - labels).abs().mean().item()
        train_mse = acc_mse / len(train_loader)
        train_mae = acc_mae / len(train_loader)

        # val real
        model.eval()
        acc_mse = acc_mae = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).view(-1)
                acc_mse += criterion(preds, labels).item()
                acc_mae += (preds - labels).abs().mean().item()
        val_mse = acc_mse / len(val_loader)
        val_mae = acc_mae / len(val_loader)

        # val synth
        acc_smae = 0.0
        with torch.no_grad():
            for imgs, labels in synth_val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).view(-1)
                acc_smae += (preds - labels).abs().mean().item()
        synth_val_mae = acc_smae / len(synth_val_loader)

        scheduler.step(val_mse)

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            early_stop_count = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            early_stop_count += 1

        lrs = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        print(f"Epoch {epoch}/{EPOCHS} | train mse={train_mse*1e4:.2f} mae={train_mae*100:.2f}% "
              f"| val_real mse={val_mse*1e4:.2f} mae={val_mae*100:.2f}% "
              f"| val_synth mae={synth_val_mae*100:.2f}% | lr={lrs}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_mse, train_mae, val_mse, val_mae, synth_val_mae])

        if early_stop_count >= EARLY_STOP:
            print(f"Early stopping — best val MAE: {best_val_mae*100:.2f}%")
            break

    json.dump({"best_val_mae_pct": round(best_val_mae * 100, 3)}, open(run_dir / "summary.json", "w"), indent=2)
    print(f"Done. Best model saved to {run_dir / 'best_model.pt'}")

    _plot_training_curves(run_dir)
    _plot_val_scatter(run_dir, device)


def _plot_val_scatter(run_dir: Path, device: torch.device):
    model = get_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False))
    model.eval()

    samples = _load_csv(_LABELS_DIR / "val.csv")
    loader  = DataLoader(ImageDataset(samples, _REAL_IMAGES_DIR), batch_size=BATCH, shuffle=False, num_workers=0)

    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds_all.append(model(imgs.to(device)).view(-1).cpu().numpy())
            gt_all.append(labels.numpy())

    preds  = np.concatenate(preds_all) * 100
    gt     = np.concatenate(gt_all)    * 100
    errors = preds - gt
    mae    = np.abs(errors).mean()

    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(gt, preds, alpha=0.7, edgecolors="none")
    lims = [min(gt.min(), preds.min()) - 2, max(gt.max(), preds.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground truth (%)"); ax.set_ylabel("Predicted (%)")
    ax.set_title(f"Val scatter — MAE {mae:.2f}%")
    fig.tight_layout()
    fig.savefig(graphs_dir / "scatter.png", dpi=150); plt.close(fig)
    print(f"Val scatter saved to {graphs_dir / 'scatter.png'}")


def _plot_training_curves(run_dir: Path):
    df = pd.read_csv(run_dir / "metrics.csv")
    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    epochs = df["epoch"]

    fig, ax = plt.subplots()
    ax.plot(epochs, df["train_mse"] * 1e4, label="train")
    ax.plot(epochs, df["val_real_mse"] * 1e4, label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE ×10⁻⁴"); ax.set_title("Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(graphs_dir / "loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, df["train_mae"] * 100, label="train")
    ax.plot(epochs, df["val_real_mae"] * 100, label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (%)"); ax.set_title("MAE")
    ax.legend(); fig.tight_layout()
    fig.savefig(graphs_dir / "mae.png", dpi=150); plt.close(fig)

    print(f"Training curves saved to {graphs_dir}")


if __name__ == "__main__":
    main()
