"""
Shadow training script for ConvNeXt-Small with synthetic data support.

Key differences from train.py:
  - Trains on real + synthetic images while backbone is frozen
  - After unfreeze, switches train loader to real-only
  - Tracks val MAE separately for real and synthetic val sets
  - Synthetic images live in data/images/raw/ alongside real ones
  - Synthetic val CSV: data/labels/synthetic_val.csv (Sample;Fineness)

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/train_convnext_synth.py
"""

import csv
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.convnext import get_convnext_small

# ── config ────────────────────────────────────────────────────────────────────
EPOCHS            = 100
LR                = 1e-3
BATCH_SIZE        = 32
NUM_WORKERS       = 4
UNFREEZE_AFTER    = 15
EARLY_STOP        = 20
SYNTH_VAL_FRAC    = 0.1   # fraction of synthetic images held out for val
SYNTH_LIMIT       = 1000  # cap total synthetic images before split (None = all)
REAL_ONLY_AFTER_UNFREEZE = False  # if False, keep mixed data after unfreeze too

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"
_SYNTH_CSV  = Path("Q:/temp/coffee_grind_gap_dataset/labels.csv")

_NORMALIZE = transforms.Normalize(
    mean=[0.1557, 0.0899, 0.0404],
    std =[0.0483, 0.0349, 0.0190],
)
TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

# ── dataset ───────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, samples, images_dir, transform=None):
        self.samples    = samples       # list of (filename, fineness_float)
        self.images_dir = Path(images_dir)
        self.transform  = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(self.images_dir / fname).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def load_csv(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    return [(row.iloc[0], float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def make_loader(samples, shuffle=True):
    return DataLoader(
        ImageDataset(samples, _IMAGES_DIR, TRANSFORM),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None,
    )


def build_loaders():
    real_train = load_csv(_LABELS_DIR / 'train.csv')
    real_val   = load_csv(_LABELS_DIR / 'val.csv')

    synth_all  = load_csv(_SYNTH_CSV)
    np.random.seed(42)
    idx        = np.random.permutation(len(synth_all))
    if SYNTH_LIMIT is not None:
        idx    = idx[:SYNTH_LIMIT]
    n_val      = max(1, int(len(idx) * SYNTH_VAL_FRAC))
    synth_val  = [synth_all[i] for i in idx[:n_val]]
    synth_train= [synth_all[i] for i in idx[n_val:]]

    mixed_train = real_train + synth_train

    print(f"Train: {len(real_train)} real + {len(synth_train)} synth = {len(mixed_train)} total")
    print(f"Val  : {len(real_val)} real | {len(synth_val)} synth")

    return (
        make_loader(mixed_train, shuffle=True),
        make_loader(real_train,  shuffle=True),   # real-only for post-unfreeze
        make_loader(real_val,    shuffle=False),
        make_loader(synth_val,   shuffle=False),
    )

# ── helpers ───────────────────────────────────────────────────────────────────

def eval_loader(model, loader, criterion, device):
    model.eval()
    acc_mse = acc_mae = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(device_type=device.type):
                preds = model(imgs).view(-1)
                acc_mse += criterion(preds, labels).item()
            acc_mae += (preds - labels).abs().mean().item()
    return acc_mse / len(loader), acc_mae / len(loader)


def save_plot(train_vals, val_vals, title, path, extra=None, extra_label=None):
    plt.figure()
    plt.plot(train_vals, label='train')
    plt.plot(val_vals,   label='val_real')
    if extra:
        plt.plot(extra, label=extra_label or 'val_synth', linestyle='--')
    plt.legend(); plt.title(title); plt.tight_layout()
    plt.savefig(path); plt.close()


def save_scatter(model, real_loader, synth_loader, device, epoch, graphs_dir, run_dir):
    model.eval()

    def collect(loader):
        preds_all, gt_all = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                preds_all.append(model(imgs).view(-1).cpu().numpy())
                gt_all.append(labels.numpy())
        return np.concatenate(preds_all) * 100, np.concatenate(gt_all) * 100

    r_preds, r_gt = collect(real_loader)
    s_preds, s_gt = collect(synth_loader)

    # combined predictions CSV
    with open(run_dir / f'predictions_epoch_{epoch:03d}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ground_truth', 'predicted', 'source'])
        w.writerows([(gt, p, 'real')  for gt, p in zip(r_gt.tolist(), r_preds.tolist())])
        w.writerows([(gt, p, 'synth') for gt, p in zip(s_gt.tolist(), s_preds.tolist())])

    # scatter plot
    plt.figure()
    plt.scatter(r_gt, r_preds, alpha=0.6, label='real',  color='saddlebrown', s=30)
    plt.scatter(s_gt, s_preds, alpha=0.6, label='synth', color='steelblue',   s=30, marker='^')
    plt.plot([0,100],[0,100],'k--',lw=1)
    plt.xlabel('Ground Truth'); plt.ylabel('Prediction')
    plt.title(f'GT vs Pred — epoch {epoch}')
    plt.legend(); plt.tight_layout()
    plt.savefig(graphs_dir / f'scatter_epoch_{epoch:03d}.png'); plt.close()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    base_dir = _ROOT / 'data' / 'runs' / 'convnext_small'
    existing = sorted(base_dir.glob('run_*'))
    run_id   = int(existing[-1].name.split('_')[1]) + 1 if existing else 1
    run_dir  = base_dir / f'run_{run_id:03d}'
    (run_dir / 'models').mkdir(parents=True, exist_ok=True)
    (run_dir / 'graphs').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Run: {run_dir.name}")

    mixed_loader, real_loader, real_val_loader, synth_val_loader = build_loaders()

    model     = get_convnext_small(freeze_backbone=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler    = GradScaler(device=device.type)

    config = {
        "model": "convnext_small", "epochs": EPOCHS, "lr": LR,
        "weight_decay": 1e-4, "batch_size": BATCH_SIZE,
        "unfreeze": False, "unfreeze_after": UNFREEZE_AFTER,
        "online_augment": False, "loss": "mse", "optimizer": "adam",
        "scheduler": "plateau", "scheduler_patience": 5,
        "adversarial": False, "adv_epsilon": None, "adv_weight": None,
        "early_stop_patience": EARLY_STOP, "num_workers": NUM_WORKERS,
        "device": device.type, "synthetic_data": str(_SYNTH_CSV),
        "synth_val_frac": SYNTH_VAL_FRAC,
        "real_only_after_unfreeze": REAL_ONLY_AFTER_UNFREEZE,
        "strategy": "mixed_frozen_then_real_only_unfrozen" if REAL_ONLY_AFTER_UNFREEZE else "mixed_throughout",
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / 'metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_mse','train_mae','val_real_mse','val_real_mae','val_synth_mse','val_synth_mae'])

    train_mses, train_maes = [], []
    real_mses,  real_maes  = [], []
    synth_mses, synth_maes = [], []
    best_val_mae   = float('inf')
    early_stop_ctr = 0
    train_loader   = mixed_loader   # start with mixed

    for epoch in range(1, EPOCHS + 1):

        # ── unfreeze: switch to real-only loader ───────────────────────────
        if UNFREEZE_AFTER is not None and epoch == UNFREEZE_AFTER + 1:
            for param in model.parameters():
                param.requires_grad = True
            backbone_params = [p for p in model.parameters()
                               if not any(p is hp for hp in optimizer.param_groups[0]['params'])]
            optimizer.add_param_group({'params': backbone_params, 'lr': LR / 10, 'weight_decay': 1e-4})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            if REAL_ONLY_AFTER_UNFREEZE:
                train_loader = real_loader
                print(f"Epoch {epoch}: backbone unfrozen, switched to real-only training, lr → {LR/10:.2e}")
            else:
                print(f"Epoch {epoch}: backbone unfrozen, keeping mixed training, lr → {LR/10:.2e}")

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        acc_mse = acc_mae = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                preds = model(imgs).view(-1)
                loss  = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                acc_mse += loss.detach().item()
                acc_mae += (preds.detach() - labels).abs().mean().item()
        train_mses.append(acc_mse / len(train_loader))
        train_maes.append(acc_mae / len(train_loader))

        # ── val ────────────────────────────────────────────────────────────
        r_mse, r_mae = eval_loader(model, real_val_loader,  criterion, device)
        s_mse, s_mae = eval_loader(model, synth_val_loader, criterion, device)
        real_mses.append(r_mse);  real_maes.append(r_mae)
        synth_mses.append(s_mse); synth_maes.append(s_mae)

        scheduler.step(r_mse)

        if r_mae < best_val_mae:
            best_val_mae   = r_mae
            early_stop_ctr = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pt')
        else:
            early_stop_ctr += 1

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train mse={train_mses[-1]*1e4:.2f} mae={train_maes[-1]*100:.2f} | "
              f"val_real mse={r_mse*1e4:.2f} mae={r_mae*100:.2f} | "
              f"val_synth mae={s_mae*100:.2f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_mses[-1], train_maes[-1], r_mse, r_mae, s_mse, s_mae])

        save_plot([x*1e4 for x in train_mses], [x*1e4 for x in real_mses], 'MSE',
                  run_dir/'graphs'/'loss.png', [x*1e4 for x in synth_mses], 'val_synth')
        save_plot([x*100 for x in train_maes], [x*100 for x in real_maes], 'MAE (%)',
                  run_dir/'graphs'/'mae.png',  [x*100 for x in synth_maes], 'val_synth')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), run_dir / 'models' / f'epoch_{epoch:03d}.pt')
            save_scatter(model, real_val_loader, synth_val_loader, device, epoch, run_dir / 'graphs', run_dir)

        if EARLY_STOP > 0 and early_stop_ctr >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch} (best real val MAE={best_val_mae*100:.2f}%)")
            break

    print(f"Done. Best real val MAE: {best_val_mae*100:.2f}%")
    print(f"Run saved to {run_dir}")


if __name__ == '__main__':
    main()
