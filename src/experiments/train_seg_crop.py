"""
Experiment: segment → tight-crop → resize to 420px, then train ConvNeXt-Small.

Pipeline per image:
  1. Brown-colour mask → background becomes pure black
  2. Bounding box of non-black pixels → tight crop
  3. Pad to square (larger dimension)
  4. Resize to TARGET px — LANCZOS
  5. ToTensor + Normalize

Usage:
    uv run python src/train_seg_crop.py
    uv run python src/train_seg_crop.py --synthetic --synth-limit 1500
    uv run python src/train_seg_crop.py --target 420
"""

import argparse, csv, gc, json, sys
from scipy.ndimage import uniform_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _NORMALIZE, load_synthetic_split, SyntheticDataset
from models.convnext import get_convnext_small

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH            = 32
NUM_WORKERS      = 4
LR_HEAD          = 1e-3
LR_BACKBONE      = 1e-4
EPOCHS           = 100
UNFREEZE_AFTER   = 15
EARLY_STOP       = 12
TARGET           = 420   # override with --target


# ── Seg-crop transform ───────────────────────────────────────────────────────

def _seg_mask(arr):
    """Brown-colour mask, works for any square image size."""
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3).astype(np.float32)
    color = np.array([160, 82, 45], dtype=np.float32)
    color /= np.linalg.norm(color)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    closeness = (flat / norms) @ color
    binary = (closeness.reshape(h, w) > closeness.mean()).astype("float")
    filter_size = max(1, round(62 * h / 1080))   # scale filter to image size
    blurred = uniform_filter(binary, filter_size)
    return (blurred > 0.95)[..., None]


class SegCropResize:
    """
    1. Apply brown-colour mask → background becomes black
    2. Find bounding box of non-black pixels
    3. Pad to square (larger dimension)
    4. Resize to `size`px
    """
    def __init__(self, size=420):
        self.size = size

    def __call__(self, pil_img):
        arr = np.asarray(pil_img.convert("RGB"))
        masked = (arr * _seg_mask(arr)).astype(np.uint8)

        # bounding box of non-black pixels
        nonblack = masked.max(axis=2) > 0
        rows = np.any(nonblack, axis=1)
        cols = np.any(nonblack, axis=0)
        if not rows.any():          # fallback: entire image is black
            masked = arr
            rows = cols = np.ones(arr.shape[0], dtype=bool)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        crop = masked[r0:r1+1, c0:c1+1]

        # pad to square using the larger side
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        sq = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - ch) // 2
        x0 = (side - cw) // 2
        sq[y0:y0+ch, x0:x0+cw] = crop

        return Image.fromarray(sq).resize((self.size, self.size), Image.LANCZOS)


def make_transform(size):
    return transforms.Compose([
        SegCropResize(size),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class RealDataset(Dataset):
    def __init__(self, csv_path, transform):
        import pandas as pd
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        self.samples   = [(row.iloc[0], float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(_IMAGES_DIR / fname).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def _lkw():
    return dict(num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                prefetch_factor=4 if NUM_WORKERS > 0 else None,
                multiprocessing_context="spawn" if NUM_WORKERS > 0 else None)


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_ep(model, loader, opt, crit, scaler, device):
    model.train(); mse = mae = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        opt.zero_grad()
        with autocast(device_type=device.type):
            preds = model(imgs).view(-1)
            loss  = crit(preds, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad():
            mse += loss.item()
            mae += (preds.detach() - labels).abs().mean().item()
    return mse / len(loader), mae / len(loader)


def _eval_ep(model, loader, device, crit):
    model.eval(); mse = mae = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                preds = model(imgs).view(-1)
                mse  += crit(preds, labels).item()
            mae += (preds - labels).abs().mean().item()
    return mse / len(loader), mae / len(loader)


def _scatter(model, loader, device, path):
    model.eval(); pa, ga = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            pa.append(model(imgs.to(device, non_blocking=True)).view(-1).cpu().numpy())
            ga.append(labels.numpy())
    p, g = np.concatenate(pa) * 100, np.concatenate(ga) * 100
    mae = np.abs(p - g).mean()
    mse = np.mean((p - g) ** 2)
    with open(Path(str(path)).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ground_truth", "predicted"])
        w.writerows(zip(g.tolist(), p.tolist()))
    fig, ax = plt.subplots()
    ax.scatter(g, p, alpha=0.5, s=20, color="saddlebrown")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Ground Truth (%)", ylabel="Prediction (%)", title=f"MAE={mae:.2f}%")
    ax.text(0.05, 0.95, f"MSE: {mse:.4f}", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _plot(train_h, val_h, ylabel, path):
    scale = 10000 if ylabel == "MSE" else 100
    fig, ax = plt.subplots()
    ax.plot([v * scale for v in train_h], label="train")
    ax.plot([v * scale for v in val_h],   label="val")
    ax.set(xlabel="Epoch", ylabel=ylabel); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def train_loop(model, train_loader, val_loader, device, out_dir):
    g = out_dir / "graphs"; g.mkdir(parents=True, exist_ok=True)
    m = out_dir / "models"; m.mkdir(parents=True, exist_ok=True)
    crit   = nn.MSELoss()
    opt    = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=LR_HEAD, weight_decay=1e-4)
    sch    = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(device=device.type)
    tr_mse, tr_mae, v_mse, v_mae = [], [], [], []
    best = float("inf"); best_ep = 0; pat = 0

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_mse", "val_mae"])

    for ep in range(1, EPOCHS + 1):
        if ep == UNFREEZE_AFTER + 1:
            for p in model.parameters(): p.requires_grad = True
            known = {id(p) for pg in opt.param_groups for p in pg["params"]}
            new_p = [p for p in model.parameters() if id(p) not in known]
            opt.add_param_group({"params": new_p, "lr": LR_BACKBONE, "weight_decay": 1e-4})
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
            print(f"  E{ep}: backbone unfrozen @ lr={LR_BACKBONE:.1e}")

        t_mse, t_mae = _train_ep(model, train_loader, opt, crit, scaler, device)
        vm, vma      = _eval_ep(model, val_loader, device, crit)
        tr_mse.append(t_mse); tr_mae.append(t_mae); v_mse.append(vm); v_mae.append(vma)
        sch.step(vm)

        if vma < best:
            best = vma; best_ep = ep; pat = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            pat += 1

        lrs = "/".join(f"{pg['lr']:.2e}" for pg in opt.param_groups)
        tag = " *" if pat == 0 else ""
        print(f"  E{ep}/{EPOCHS} | train mae={t_mae*100:.2f}% | "
              f"val mae={vma*100:.2f}% | lr={lrs}{tag}")

        with open(out_dir / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep, t_mse, t_mae, vm, vma])

        _plot(tr_mse, v_mse, "MSE",     g / "mse.png")
        _plot(tr_mae, v_mae, "MAE (%)", g / "mae.png")

        if ep % 5 == 0 or ep == EPOCHS:
            torch.save(model.state_dict(), m / f"epoch_{ep:03d}.pt")
            _scatter(model, val_loader, device, g / f"scatter_e{ep:03d}.png")

        if pat >= EARLY_STOP:
            print(f"  Early stop @ E{ep} (best={best*100:.2f}% @ E{best_ep})")
            break

    return {"best_val_mae": best, "best_epoch": best_ep, "epochs_run": ep}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int,   default=TARGET,
                        help=f"resize target after seg-crop (default: {TARGET})")
    parser.add_argument("--synthetic",   action="store_true",
                        help="mix in synthetic images from data/images/synthetic/")
    parser.add_argument("--synth-limit", type=int,   default=None, metavar="N",
                        help="cap number of synthetic train images (default: all)")
    parser.add_argument("--synth-frac",  type=float, default=0.1,  metavar="F",
                        help="fraction of synthetic data held out for val (default: 0.1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base = _ROOT / "data" / "runs" / "seg_crop"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))
    run_id   = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir  = base / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Target: {args.target}px  |  Run: {run_dir}\n")

    tf = make_transform(args.target)
    train_ds = RealDataset(_LABELS_DIR / "train.csv", tf)
    val_ds   = RealDataset(_LABELS_DIR / "val.csv",   tf)

    if args.synthetic:
        synth_train, synth_val = load_synthetic_split(args.synth_frac)
        if args.synth_limit is not None:
            synth_train = synth_train[:args.synth_limit]
        synth_train_ds = SyntheticDataset(synth_train, transform=tf)
        mixed_ds = ConcatDataset([train_ds, synth_train_ds])
        print(f"Real: {len(train_ds)} train + {len(val_ds)} val  "
              f"Synth: {len(synth_train)} train + {len(synth_val)} val  "
              f"| total train: {len(mixed_ds)}")
        train_loader = DataLoader(mixed_ds, batch_size=BATCH, shuffle=True, **_lkw())
    else:
        print(f"Real: {len(train_ds)} train + {len(val_ds)} val")
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, **_lkw())

    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, **_lkw())

    json.dump({
        "target_size": args.target, "epochs": EPOCHS,
        "unfreeze_after": UNFREEZE_AFTER, "early_stop": EARLY_STOP,
        "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE, "batch": BATCH,
        "synthetic": args.synthetic, "synth_limit": args.synth_limit,
    }, open(run_dir / "config.json", "w"), indent=2)

    model = get_convnext_small(freeze_backbone=True).to(device)
    h = train_loop(model, train_loader, val_loader, device, run_dir)

    summary = {
        "best_val_mae_pct": round(h["best_val_mae"] * 100, 3),
        "best_epoch": h["best_epoch"],
        "epochs_run": h["epochs_run"],
        "target_size": args.target,
        "synthetic": args.synthetic,
        "synth_limit": args.synth_limit,
    }
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"Best val MAE: {h['best_val_mae']*100:.2f}%  @ epoch {h['best_epoch']}")
    print(f"Saved → {run_dir}")


if __name__ == "__main__":
    main()
