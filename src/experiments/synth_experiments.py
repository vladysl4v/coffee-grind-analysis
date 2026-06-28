"""
Synthetic data experiments — runs everything in one shot.

Each experiment uses ONE synth dataset at a time (never combined).

ConvNeXt-Small:
  cnx_new_A   Pretrain on new_dataset (1500) → head tune → full tune on real
  cnx_new_B   Mixed: real + new_dataset (1500)
  cnx_gap_A   Pretrain on gap_dataset (1500) → head tune → full tune on real
  cnx_gap_B   Mixed: real + gap_dataset (1500)
  cnx_C       Scaling sweep on gap_dataset (most images): 100 → max, step 400

ViT-Base:
  vit_new_A   Pretrain on new_dataset (1500) → fine-tune on real
  vit_new_B   Mixed: real + new_dataset (1500)
  vit_gap_A   Pretrain on gap_dataset (1500) → fine-tune on real
  vit_gap_B   Mixed: real + gap_dataset (1500)

Output: data/runs/synth_experiments/run_XXX/

Usage:
    uv run python src/synth_experiments.py                       # all 9
    uv run python src/synth_experiments.py --skip cnx_C         # skip scaling
    uv run python src/synth_experiments.py --only vit_new_A     # one experiment
"""

import argparse, csv, gc, json, shutil, sys, traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_loaders, DEFAULT_EVAL_TRANSFORM, _NORMALIZE
from models.convnext import get_convnext_small
from models.vit import get_vit

_ROOT    = Path(__file__).parent.parent
NEW_DIR  = Path("Q:/temp/coffee_grind_new_dataset")
GAP_DIR  = Path("Q:/temp/coffee_grind_gap_dataset")

# ── Hyperparameters ─────────────────────────────────────────────────────────────
BATCH               = 32
VIT_BATCH           = 16   # ViT-Base unfrozen exceeds 12 GB at batch 32
NUM_WORKERS         = 4
LR_HEAD             = 1e-3
LR_BACKBONE         = 1e-4
LR_FINETUNE_FULL    = 1e-5

PRETRAIN_EPOCHS      = 60
PRETRAIN_UNFREEZE    = 15
FINETUNE_HEAD_EPOCHS = 20
FINETUNE_FULL_EPOCHS = 20
MIXED_EPOCHS         = 100
MIXED_UNFREEZE       = 15
SCALING_EPOCHS       = 25   # per-step cap; early stop at 10 cuts it shorter
EARLY_STOP           = 12
SYNTH_TOTAL          = 1500
SCALE_STEP           = 400

def _make_transforms(crop):
    train = transforms.Compose([transforms.CenterCrop(crop), transforms.ToTensor(), _NORMALIZE])
    val   = transforms.Compose([transforms.CenterCrop(crop), transforms.ToTensor(), _NORMALIZE])
    return train, val

TRAIN_TRANSFORM, EVAL_TRANSFORM = _make_transforms(224)


# ── Data ────────────────────────────────────────────────────────────────────────

class ExternalSynthDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples   # list of (Path, label_0_1)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def _read_dir(dir_path, seed=0):
    df = pd.read_csv(dir_path / "labels.csv", sep=";", decimal=",")
    s  = [(dir_path / str(r.iloc[0]), float(r.iloc[1]) / 100.0) for _, r in df.iterrows()]
    np.random.default_rng(seed).shuffle(s)
    return s


def synth_train(dir_path, n=SYNTH_TOTAL, seed=42):
    return _read_dir(dir_path, seed=seed)[:n]


def synth_val(dir_path, frac=0.1, seed=42):
    s = _read_dir(dir_path, seed=seed)
    return s[:max(1, int(len(s) * frac))]


def _lkw():
    return dict(num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                prefetch_factor=4 if NUM_WORKERS > 0 else None,
                multiprocessing_context="spawn" if NUM_WORKERS > 0 else None)


def train_dl(samples, batch=BATCH):
    return DataLoader(ExternalSynthDataset(samples, TRAIN_TRANSFORM),
                      batch_size=batch, shuffle=True, **_lkw())


def eval_dl(samples, batch=BATCH):
    return DataLoader(ExternalSynthDataset(samples, EVAL_TRANSFORM),
                      batch_size=batch, shuffle=False, **_lkw())


def mixed_dl(real_ds, synth_samples, batch=BATCH):
    ds = ConcatDataset([real_ds, ExternalSynthDataset(synth_samples, TRAIN_TRANSFORM)])
    return DataLoader(ds, batch_size=batch, shuffle=True, **_lkw())


def real_loaders():
    return get_loaders(batch_size=BATCH, num_workers=NUM_WORKERS,
                       train_transform=TRAIN_TRANSFORM,
                       eval_transform=EVAL_TRANSFORM)


# ── Training core ────────────────────────────────────────────────────────────────

def _train_ep(model, loader, optimizer, criterion, scaler, device):
    model.train(); mse = mae = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            preds = model(imgs).view(-1)
            loss  = criterion(preds, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            mse += loss.item()
            mae += (preds.detach() - labels).abs().mean().item()
    return mse / len(loader), mae / len(loader)


def _eval_ep(model, loader, device, criterion):
    model.eval(); mse = mae = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                preds = model(imgs).view(-1)
                mse  += criterion(preds, labels).item()
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

    csv_path = Path(str(path)).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "predicted"])
        w.writerows(zip(g.tolist(), p.tolist()))

    fig, ax = plt.subplots()
    ax.scatter(g, p, alpha=0.5, s=20, color="saddlebrown")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Ground Truth (%)", ylabel="Prediction (%)", title=f"MAE={mae:.2f}%")
    ax.text(0.05, 0.95, f"MSE: {mse:.4f}", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return float(mae)


def _plot(train_h, val_h, ylabel, path):
    scale = 10000 if ylabel == "MSE" else 100
    fig, ax = plt.subplots()
    ax.plot([v * scale for v in train_h], label="train")
    ax.plot([v * scale for v in val_h],   label="val")
    ax.set(xlabel="Epoch", ylabel=ylabel); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def train_loop(model, train_loader, val_loader, epochs, lr, device,
               out_dir, prefix, unfreeze_after=None, unfreeze_lr=LR_BACKBONE,
               early_stop=EARLY_STOP, scatter_every=5):
    g = out_dir / "graphs"; g.mkdir(parents=True, exist_ok=True)
    m = out_dir / "models"; m.mkdir(parents=True, exist_ok=True)
    crit   = nn.MSELoss()
    opt    = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=lr, weight_decay=1e-4)
    sch    = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(device=device.type)
    tr_mse, tr_mae, v_mse, v_mae = [], [], [], []
    best = float("inf"); best_ep = 0; pat = 0; last_ep = 1

    with open(out_dir / f"{prefix}_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_mae", "val_mse", "val_mae"])

    for ep in range(1, epochs + 1):
        last_ep = ep
        if unfreeze_after and ep == unfreeze_after + 1:
            for p in model.parameters(): p.requires_grad = True
            known = {id(p) for pg in opt.param_groups for p in pg["params"]}
            new_p = [p for p in model.parameters() if id(p) not in known]
            opt.add_param_group({"params": new_p, "lr": unfreeze_lr, "weight_decay": 1e-4})
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
            print(f"  [{prefix}] E{ep}: backbone unfrozen @ lr={unfreeze_lr:.1e}")

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
        print(f"  [{prefix}] E{ep}/{epochs} | train mae={t_mae*100:.2f}% | "
              f"val mae={vma*100:.2f}% | lr={opt.param_groups[0]['lr']:.1e}{tag}")

        with open(out_dir / f"{prefix}_metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep, t_mse, t_mae, vm, vma])

        _plot(tr_mse, v_mse, "MSE",     g / f"{prefix}_mse.png")
        _plot(tr_mae, v_mae, "MAE (%)", g / f"{prefix}_mae.png")

        if ep % 5 == 0 or ep == epochs:
            torch.save(model.state_dict(), m / f"{prefix}_epoch_{ep:03d}.pt")

        if ep % scatter_every == 0 or ep == epochs:
            _scatter(model, val_loader, device, g / f"{prefix}_scatter_e{ep:03d}.png")

        if early_stop > 0 and pat >= early_stop:
            print(f"  [{prefix}] Early stop @ E{ep} (best={best*100:.2f}% @ E{best_ep})")
            break

    return {"best_val_mae": best, "best_epoch": best_ep, "epochs_run": last_ep}


# ── Generic experiment runners ───────────────────────────────────────────────────

_HPARAMS = {
    "BATCH": BATCH, "LR_HEAD": LR_HEAD, "LR_BACKBONE": LR_BACKBONE,
    "LR_FINETUNE_FULL": LR_FINETUNE_FULL,
    "PRETRAIN_EPOCHS": PRETRAIN_EPOCHS, "PRETRAIN_UNFREEZE": PRETRAIN_UNFREEZE,
    "FINETUNE_HEAD_EPOCHS": FINETUNE_HEAD_EPOCHS, "FINETUNE_FULL_EPOCHS": FINETUNE_FULL_EPOCHS,
    "MIXED_EPOCHS": MIXED_EPOCHS, "MIXED_UNFREEZE": MIXED_UNFREEZE,
    "SCALING_EPOCHS": SCALING_EPOCHS, "EARLY_STOP": EARLY_STOP,
    "SYNTH_TOTAL": SYNTH_TOTAL, "SCALE_STEP": SCALE_STEP,
}

def _save_config(out_dir, name, extra=None):
    cfg = {"experiment": name, **_HPARAMS, **(extra or {})}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _load_if_done(out_dir):
    p = out_dir / "summary.json"
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        print(f"  Already done — skipping ({p})")
        return s
    return None


def _gpu_free():
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  GPU cleared — {free:.1f} GB free\n")


def _run_safe(name, fn, summaries):
    try:
        summaries[name] = fn()
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n!! {name} — CUDA OUT OF MEMORY: {e}")
        print("!! Skipping this experiment. Free GPU memory and re-run with --only to retry.")
        summaries[name] = {"experiment": name, "error": "OOM", "detail": str(e)}
    except Exception as e:
        print(f"\n!! {name} — FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        summaries[name] = {"experiment": name, "error": type(e).__name__, "detail": str(e)}
    finally:
        _gpu_free()


def run_pretrain_finetune(device, out_dir, val_loader, real_train_loader,
                           s_train, s_val, model_fn, name, batch=BATCH):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_if_done(out_dir)
    if done: return done

    _save_config(out_dir, name, {"synth_train_n": len(s_train), "synth_val_n": len(s_val),
                                  "batch": batch, "mode": "pretrain_finetune"})
    print(f"  Synth train: {len(s_train)}  Synth val: {len(s_val)}  Batch: {batch}")

    # Phase 1 — pretrain on synth
    print(f"\n  Phase 1: pretrain on synth ({PRETRAIN_EPOCHS} ep, unfreeze@{PRETRAIN_UNFREEZE})")
    model = model_fn().to(device)
    h1 = train_loop(model, train_dl(s_train, batch), eval_dl(s_val, batch),
                    PRETRAIN_EPOCHS, LR_HEAD, device, out_dir, "pretrain",
                    unfreeze_after=PRETRAIN_UNFREEZE, unfreeze_lr=LR_BACKBONE)

    model.load_state_dict(torch.load(out_dir / "pretrain_best.pt", weights_only=False))
    _, mae_no_ft = _eval_ep(model, val_loader, device, nn.MSELoss())
    print(f"  Real val MAE (no fine-tune): {mae_no_ft*100:.2f}%")

    # Phase 2a — head only on real
    print(f"\n  Phase 2a: head fine-tune on real ({FINETUNE_HEAD_EPOCHS} ep)")
    model = model_fn().to(device)
    model.load_state_dict(torch.load(out_dir / "pretrain_best.pt", weights_only=False))
    h2a = train_loop(model, real_train_loader, val_loader,
                     FINETUNE_HEAD_EPOCHS, LR_HEAD, device, out_dir, "finetune_head",
                     early_stop=EARLY_STOP)

    # Phase 2b — full model on real at very low LR
    print(f"\n  Phase 2b: full model fine-tune on real ({FINETUNE_FULL_EPOCHS} ep)")
    model.load_state_dict(torch.load(out_dir / "finetune_head_best.pt", weights_only=False))
    for p in model.parameters(): p.requires_grad = True
    h2b = train_loop(model, real_train_loader, val_loader,
                     FINETUNE_FULL_EPOCHS, LR_FINETUNE_FULL, device, out_dir, "finetune_full",
                     early_stop=EARLY_STOP)

    shutil.copy(out_dir / "finetune_full_best.pt", out_dir / "final_best.pt")
    model.load_state_dict(torch.load(out_dir / "final_best.pt", weights_only=False))
    _, mae_final = _eval_ep(model, val_loader, device, nn.MSELoss())

    summary = {
        "experiment": name,
        "synth_train_n": len(s_train),
        "real_val_mae_pretrain_only_pct":       round(mae_no_ft * 100, 3),
        "real_val_mae_after_head_finetune_pct": round(h2a["best_val_mae"] * 100, 3),
        "real_val_mae_after_full_finetune_pct": round(mae_final * 100, 3),
        "pretrain_synth_mae_pct":  round(h1["best_val_mae"] * 100, 3),
        "pretrain_epochs_run":     h1["epochs_run"],
        "head_ft_epochs_run":      h2a["epochs_run"],
        "full_ft_epochs_run":      h2b["epochs_run"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  {name}: {mae_no_ft*100:.2f}% → {h2a['best_val_mae']*100:.2f}% → {mae_final*100:.2f}%")
    return summary


def run_mixed(device, out_dir, val_loader, real_train_loader, s_train, model_fn, name, batch=BATCH):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_if_done(out_dir)
    if done: return done

    total = len(real_train_loader.dataset) + len(s_train)
    _save_config(out_dir, name, {"synth_n": len(s_train), "real_n": len(real_train_loader.dataset),
                                  "total_n": total, "batch": batch, "mode": "mixed"})
    print(f"  Real: {len(real_train_loader.dataset)}  Synth: {len(s_train)}  Total: {total}  Batch: {batch}")

    model = model_fn().to(device)
    h = train_loop(model, mixed_dl(real_train_loader.dataset, s_train, batch), val_loader,
                   MIXED_EPOCHS, LR_HEAD, device, out_dir, "mixed",
                   unfreeze_after=MIXED_UNFREEZE, unfreeze_lr=LR_BACKBONE,
                   early_stop=EARLY_STOP)

    shutil.copy(out_dir / "mixed_best.pt", out_dir / "final_best.pt")
    summary = {"experiment": name, "synth_n": len(s_train),
               "best_val_mae_pct": round(h["best_val_mae"] * 100, 3),
               "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"]}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  {name}: best real val MAE = {h['best_val_mae']*100:.2f}%")
    return summary


def _draw_scaling_curve(results, out_path):
    ns   = [r["n_synth"]         for r in results]
    maes = [r["best_val_mae_pct"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ns, maes, marker="o", color="saddlebrown", linewidth=2, markersize=7)
    for n, m in zip(ns, maes):
        ax.annotate(f"{m:.1f}%", (n, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set(xlabel="Synthetic images added (gap_dataset)",
           ylabel="Best real val MAE (%)",
           title="ConvNeXt: real val MAE vs. synthetic data quantity")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def run_scaling(device, out_dir, val_loader, real_train_loader):
    """Scaling sweep on gap_dataset (5000 images — most range)."""
    print(f"\n{'='*60}\ncnx_C: ConvNeXt Scaling (gap_dataset)\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_if_done(out_dir)
    if done: return done

    all_synth = _read_dir(GAP_DIR)
    max_n     = len(all_synth)
    steps     = sorted(set(list(range(100, max_n, SCALE_STEP)) + [max_n]))
    _save_config(out_dir, "cnx_C_scaling", {"dataset": "gap_dataset", "max_synth": max_n,
                                             "steps": steps, "mode": "scaling"})
    print(f"  Gap dataset: {max_n} images  Steps ({len(steps)}): {steps}")

    csv_path = out_dir / "scaling_results.csv"
    results, completed_ns = [], set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed_ns.add(int(row["n_synth"]))
                results.append({"n_synth": int(row["n_synth"]),
                                 "best_val_mae_pct": float(row["best_val_mae_pct"]),
                                 "best_epoch": int(row["best_epoch"]),
                                 "epochs_run": int(row["epochs_run"])})
        print(f"  Resuming: {len(completed_ns)} steps already done")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["n_synth", "best_val_mae_pct", "best_epoch", "epochs_run"])

    for n in steps:
        if n in completed_ns:
            print(f"  N={n:5d} → already done, skipping")
            continue
        print(f"\n  ── N={n} ──")
        step_dir = out_dir / f"scale_{n:05d}"; step_dir.mkdir(exist_ok=True)
        model = get_convnext_small(freeze_backbone=True).to(device)
        h = train_loop(model, mixed_dl(real_train_loader.dataset, all_synth[:n]),
                       val_loader, SCALING_EPOCHS, LR_HEAD, device, step_dir, f"n{n:05d}",
                       unfreeze_after=15, unfreeze_lr=LR_BACKBONE,
                       early_stop=EARLY_STOP, scatter_every=SCALING_EPOCHS)
        row = {"n_synth": n, "best_val_mae_pct": round(h["best_val_mae"] * 100, 3),
               "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"]}
        results.append(row)
        del model; _gpu_free()
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row["n_synth"], row["best_val_mae_pct"],
                                    row["best_epoch"], row["epochs_run"]])
        print(f"  N={n:5d} → {row['best_val_mae_pct']:.2f}%")
        _draw_scaling_curve(results, out_dir / "scaling_curve.png")

    summary = {"experiment": "cnx_C_scaling", "dataset": "gap_dataset",
               "max_synth": max_n, "steps": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Scaling done → {csv_path}")
    return summary


# ── Entry point ──────────────────────────────────────────────────────────────────

ALL_EXPS = ["cnx_new_A", "cnx_new_B", "cnx_gap_A", "cnx_gap_B", "cnx_C",
            "vit_new_A", "vit_new_B", "vit_gap_A", "vit_gap_B"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only",  nargs="+", choices=ALL_EXPS, metavar="EXP")
    parser.add_argument("--skip",  nargs="+", choices=ALL_EXPS, metavar="EXP")
    parser.add_argument("--fresh", action="store_true",
                        help="always create a new run dir instead of resuming the latest incomplete one")
    parser.add_argument("--crop",  type=int, default=224,
                        help="centre-crop size for train and val (default: 224)")
    args = parser.parse_args()

    global TRAIN_TRANSFORM, EVAL_TRANSFORM
    TRAIN_TRANSFORM, EVAL_TRANSFORM = _make_transforms(args.crop)

    to_run = set(args.only or ALL_EXPS) - set(args.skip or [])
    to_run = [e for e in ALL_EXPS if e in to_run]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  Crop: {args.crop}px  |  Running: {to_run}\n")

    base = _ROOT / "data" / "runs" / "synth_experiments"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))

    if not args.fresh and existing:
        latest = existing[-1]
        if not (latest / "all_summaries.json").exists():
            run_dir = latest
            print(f"Resuming incomplete run → {run_dir}\n")
        else:
            run_id  = int(existing[-1].name.split("_")[1]) + 1
            run_dir = base / f"run_{run_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Run dir: {run_dir}\n")
    else:
        run_id  = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
        run_dir = base / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run dir: {run_dir}\n")

    real_train_ldr, val_ldr, _ = real_loaders()
    print(f"Real — train: {len(real_train_ldr.dataset)}  val: {len(val_ldr.dataset)}\n")

    cnx = lambda: get_convnext_small(freeze_backbone=True)
    vit = lambda: get_vit(freeze_backbone=True)

    summaries = {}

    if "cnx_new_A" in to_run:
        _run_safe("cnx_new_A", lambda: run_pretrain_finetune(
            device, run_dir / "cnx_new_A", val_ldr, real_train_ldr,
            s_train=synth_train(NEW_DIR), s_val=synth_val(NEW_DIR),
            model_fn=cnx, name="cnx_new_A | ConvNeXt | new_dataset | pretrain→finetune"),
            summaries)

    if "cnx_new_B" in to_run:
        _run_safe("cnx_new_B", lambda: run_mixed(
            device, run_dir / "cnx_new_B", val_ldr, real_train_ldr,
            s_train=synth_train(NEW_DIR), model_fn=cnx,
            name="cnx_new_B | ConvNeXt | new_dataset | mixed"),
            summaries)

    if "cnx_gap_A" in to_run:
        _run_safe("cnx_gap_A", lambda: run_pretrain_finetune(
            device, run_dir / "cnx_gap_A", val_ldr, real_train_ldr,
            s_train=synth_train(GAP_DIR), s_val=synth_val(GAP_DIR),
            model_fn=cnx, name="cnx_gap_A | ConvNeXt | gap_dataset | pretrain→finetune"),
            summaries)

    if "cnx_gap_B" in to_run:
        _run_safe("cnx_gap_B", lambda: run_mixed(
            device, run_dir / "cnx_gap_B", val_ldr, real_train_ldr,
            s_train=synth_train(GAP_DIR), model_fn=cnx,
            name="cnx_gap_B | ConvNeXt | gap_dataset | mixed"),
            summaries)

    if "cnx_C" in to_run:
        _run_safe("cnx_C", lambda: run_scaling(
            device, run_dir / "cnx_C", val_ldr, real_train_ldr),
            summaries)

    if "vit_new_A" in to_run:
        _run_safe("vit_new_A", lambda: run_pretrain_finetune(
            device, run_dir / "vit_new_A", val_ldr, real_train_ldr,
            s_train=synth_train(NEW_DIR), s_val=synth_val(NEW_DIR),
            model_fn=vit, name="vit_new_A | ViT | new_dataset | pretrain→finetune",
            batch=VIT_BATCH),
            summaries)

    if "vit_new_B" in to_run:
        _run_safe("vit_new_B", lambda: run_mixed(
            device, run_dir / "vit_new_B", val_ldr, real_train_ldr,
            s_train=synth_train(NEW_DIR), model_fn=vit,
            name="vit_new_B | ViT | new_dataset | mixed",
            batch=VIT_BATCH),
            summaries)

    if "vit_gap_A" in to_run:
        _run_safe("vit_gap_A", lambda: run_pretrain_finetune(
            device, run_dir / "vit_gap_A", val_ldr, real_train_ldr,
            s_train=synth_train(GAP_DIR), s_val=synth_val(GAP_DIR),
            model_fn=vit, name="vit_gap_A | ViT | gap_dataset | pretrain→finetune",
            batch=VIT_BATCH),
            summaries)

    if "vit_gap_B" in to_run:
        _run_safe("vit_gap_B", lambda: run_mixed(
            device, run_dir / "vit_gap_B", val_ldr, real_train_ldr,
            s_train=synth_train(GAP_DIR), model_fn=vit,
            name="vit_gap_B | ViT | gap_dataset | mixed",
            batch=VIT_BATCH),
            summaries)

    with open(run_dir / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'='*60}\nAll done → {run_dir}\n{'='*60}")
    print(f"\n{'Experiment':<16} {'Result':>45}")
    print("-" * 63)
    for k, s in summaries.items():
        if "error" in s:
            val = f"FAILED: {s['error']}"
        elif "real_val_mae_after_full_finetune_pct" in s:
            val = (f"{s['real_val_mae_pretrain_only_pct']:.2f}% → "
                   f"{s['real_val_mae_after_head_finetune_pct']:.2f}% → "
                   f"{s['real_val_mae_after_full_finetune_pct']:.2f}%")
        elif "best_val_mae_pct" in s:
            val = f"best {s['best_val_mae_pct']:.2f}% @ epoch {s['best_epoch']}"
        elif "steps" in s and s["steps"]:
            best = min(s["steps"], key=lambda x: x["best_val_mae_pct"])
            val = f"best {best['best_val_mae_pct']:.2f}% @ N={best['n_synth']}"
        else:
            val = "no results"
        print(f"{k:<16} {val:>45}")


if __name__ == "__main__":
    main()
