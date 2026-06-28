"""
Fine-tune a saved synth-pretrained ConvNeXt-Small on real images.
Loads pretrain_best.pt from a previous synth_experiments run and fine-tunes
with no hard epoch cap — early stop is the only termination condition.

Phases:
  2a. Head-only fine-tune on real  (early stop, patience=15)
  2b. Full model fine-tune on real (early stop, patience=15, lr=1e-5)

Usage:
    uv run python src/train_pretrain_finetune.py
    uv run python src/train_pretrain_finetune.py --checkpoint data/runs/synth_experiments/run_001/cnx_new_A/pretrain_best.pt
"""

import argparse, csv, json, shutil, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_loaders, _NORMALIZE
from models.convnext import get_convnext_small
from torchvision import transforms

_ROOT = Path(__file__).parent.parent

_DEFAULT_CHECKPOINT = (
    _ROOT / "data/runs/synth_experiments/run_001/cnx_new_A/pretrain_best.pt"
)

# ── Hyperparameters ──────────────────────────────────────────────────────────────
BATCH       = 32
NUM_WORKERS = 4
LR_HEAD     = 1e-3
LR_FULL     = 1e-5
EARLY_STOP  = 15   # more patience since there's no epoch cap

TRAIN_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

# ── Training helpers ──────────────────────────────────────────────────────────────

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


def train_loop(model, train_loader, val_loader, lr, device, out_dir, prefix):
    g = out_dir / "graphs"; g.mkdir(parents=True, exist_ok=True)
    m = out_dir / "models"; m.mkdir(parents=True, exist_ok=True)
    crit   = nn.MSELoss()
    opt    = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=lr, weight_decay=1e-4)
    sch    = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(device=device.type)
    tr_mse, tr_mae, v_mse, v_mae = [], [], [], []
    best = float("inf"); best_ep = 0; pat = 0

    with open(out_dir / f"{prefix}_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_mse", "val_mae"])

    ep = 0
    while True:
        ep += 1
        t_mse, t_mae = _train_ep(model, train_loader, opt, crit, scaler, device)
        vm, vma      = _eval_ep(model, val_loader, device, crit)
        tr_mse.append(t_mse); tr_mae.append(t_mae); v_mse.append(vm); v_mae.append(vma)
        sch.step(vm)

        if vma < best:
            best = vma; best_ep = ep; pat = 0
            torch.save(model.state_dict(), out_dir / f"{prefix}_best.pt")
        else:
            pat += 1

        tag = " *" if pat == 0 else ""
        print(f"  [{prefix}] E{ep} | train mae={t_mae*100:.2f}% | "
              f"val mae={vma*100:.2f}% | lr={opt.param_groups[0]['lr']:.1e}{tag}")

        with open(out_dir / f"{prefix}_metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep, t_mse, t_mae, vm, vma])

        _plot(tr_mse, v_mse, "MSE",     g / f"{prefix}_mse.png")
        _plot(tr_mae, v_mae, "MAE (%)", g / f"{prefix}_mae.png")

        if ep % 5 == 0:
            torch.save(model.state_dict(), m / f"{prefix}_epoch_{ep:03d}.pt")
            _scatter(model, val_loader, device, g / f"{prefix}_scatter_e{ep:03d}.png")

        if pat >= EARLY_STOP:
            print(f"  [{prefix}] Early stop @ E{ep} (best={best*100:.2f}% @ E{best_ep})")
            break

    return {"best_val_mae": best, "best_epoch": best_ep, "epochs_run": ep}


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT,
                        help="path to pretrain_best.pt")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    base = _ROOT / "data" / "runs" / "pretrain_finetune"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))
    run_id   = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir  = base / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}\n")

    real_train_ldr, val_ldr, _ = get_loaders(batch_size=BATCH, num_workers=NUM_WORKERS,
                                              train_transform=TRAIN_TRANSFORM)
    print(f"Real train: {len(real_train_ldr.dataset)}  val: {len(val_ldr.dataset)}\n")

    with open(run_dir / "config.json", "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "batch": BATCH,
                   "lr_head": LR_HEAD, "lr_full": LR_FULL,
                   "early_stop_patience": EARLY_STOP, "ft_epoch_cap": "none"}, f, indent=2)

    crit = nn.MSELoss()

    # ── Phase 2a: head-only fine-tune ────────────────────────────────────────
    print("Phase 2a: head fine-tune on real (early stop only)")
    model = get_convnext_small(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
    _, mae_pretrain = _eval_ep(model, val_ldr, device, crit)
    print(f"  Starting real val MAE (pretrained weights): {mae_pretrain*100:.2f}%\n")

    h2a = train_loop(model, real_train_ldr, val_ldr, LR_HEAD, device, run_dir, "finetune_head")

    # ── Phase 2b: full model fine-tune ───────────────────────────────────────
    print("\nPhase 2b: full model fine-tune on real (early stop only, lr=1e-5)")
    model.load_state_dict(torch.load(run_dir / "finetune_head_best.pt", weights_only=False))
    for p in model.parameters(): p.requires_grad = True
    h2b = train_loop(model, real_train_ldr, val_ldr, LR_FULL, device, run_dir, "finetune_full")

    shutil.copy(run_dir / "finetune_full_best.pt", run_dir / "final_best.pt")
    model.load_state_dict(torch.load(run_dir / "final_best.pt", weights_only=False))
    _, mae_final = _eval_ep(model, val_ldr, device, crit)

    summary = {
        "checkpoint": str(args.checkpoint),
        "real_val_mae_from_checkpoint_pct":     round(mae_pretrain * 100, 3),
        "real_val_mae_after_head_finetune_pct": round(h2a["best_val_mae"] * 100, 3),
        "real_val_mae_after_full_finetune_pct": round(mae_final * 100, 3),
        "head_ft_epochs_run": h2a["epochs_run"],
        "full_ft_epochs_run": h2b["epochs_run"],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Checkpoint MAE : {mae_pretrain*100:.2f}%")
    print(f"After head FT  : {h2a['best_val_mae']*100:.2f}%  ({h2a['epochs_run']} ep)")
    print(f"After full FT  : {mae_final*100:.2f}%  ({h2b['epochs_run']} ep)")
    print(f"{'='*50}")
    print(f"Saved → {run_dir}")


if __name__ == "__main__":
    main()
