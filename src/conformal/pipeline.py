"""Load a trained run, calibrate split conformal on val, evaluate on test."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from .constants import FINENESS_SCALE
from .metrics import conformal_metrics, point_metrics
from .split import SplitConformalRegressor

from data_loader import CoffeeDataset, _CSV, _IMAGES_DIR
from models.convnext import get_convnext_small

logger = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_run_dir(run_dir: Path | None) -> Path:
    """Prefer explicit ``--run-dir``, then common run ids, else latest ``run_*``."""

    root = project_root()
    if run_dir is not None:
        p = Path(run_dir)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            logger.info("Using run directory %s", p)
            return p
        raise FileNotFoundError(f"Run directory not found: {p}")

    base = root / "data" / "runs" / "convnext_small"
    for name in ("run_001", "run001", "run_002"):
        candidate = base / name
        if candidate.is_dir():
            logger.info("Using default run directory %s", candidate.resolve())
            return candidate.resolve()

    if not base.is_dir():
        raise FileNotFoundError(
            f"No runs under {base}. Train convnext_small first or pass --run-dir."
        )
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directories in {base}")
    chosen = runs[-1].resolve()
    logger.info("Using latest run directory %s", chosen)
    return chosen


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = (project_root() / p).resolve()
        if p.is_file():
            logger.info("Using checkpoint %s", p)
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"No models/ under {run_dir}")
    cpts = sorted(models_dir.glob("epoch_*.pt"))
    if not cpts:
        raise FileNotFoundError(f"No epoch_*.pt in {models_dir}")
    ckpt = cpts[-1]
    logger.info("Using latest checkpoint %s", ckpt)
    return ckpt


def load_training_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — cannot verify baseline settings.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def assert_convnext_baseline(config: dict[str, Any]) -> None:
    if config.get("model") != "convnext_small":
        raise ValueError(
            f"Expected model 'convnext_small' in config.json, got {config.get('model')!r}. "
            "Conformal pipeline is wired for the ConvNeXt-Small baseline."
        )
    if not (config.get("unfreeze_after") is not None or config.get("unfreeze")):
        raise ValueError(
            "config.json must indicate backbone training (unfreeze or unfreeze_after) "
            "to match the adversarial ConvNeXt baseline."
        )
    if not config.get("adversarial"):
        raise ValueError(
            "config.json must have adversarial: true for the requested baseline parity."
        )


def build_convnext_inference() -> nn.Module:
    """ConvNeXt-Small with all parameters instantiated (matches full checkpoints)."""

    return get_convnext_small(freeze_backbone=False)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(images).view(-1)
        ys.append(labels.detach().cpu().numpy())
        ps.append(out.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def make_eval_loader(
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    use_augmented_data: bool = False,
    use_augmented_raw: bool = False,
    use_raw: bool = False,
) -> DataLoader:
    from data_loader import (
        _AUG_CSV,
        _AUG_IMAGES_DIR,
        _RAW_AUG_CSV,
        _RAW_AUG_IMAGES_DIR,
        _RAW_IMAGES_DIR,
    )

    if use_augmented_raw:
        csv_map, img_dir = _RAW_AUG_CSV, _RAW_AUG_IMAGES_DIR
    elif use_augmented_data:
        csv_map, img_dir = _AUG_CSV, _AUG_IMAGES_DIR
    elif use_raw:
        csv_map, img_dir = _CSV, _RAW_IMAGES_DIR
    else:
        csv_map, img_dir = _CSV, _IMAGES_DIR

    from data_loader import MASK_CROP, build_normalize
    from torchvision import transforms

    eval_transform = transforms.Compose([
        MASK_CROP,
        transforms.ToTensor(),
        build_normalize(
            use_augmented_data=use_augmented_data,
            use_augmented_raw=use_augmented_raw,
            use_raw=use_raw,
        ),
    ])
    ds = CoffeeDataset(split, transform=eval_transform, csv_map=csv_map, images_dir=img_dir)
    cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, **kwargs)


def run_split_conformal_eval(
    *,
    run_dir: Path | None,
    checkpoint: Path | None,
    alpha: float,
    batch_size: int,
    num_workers: int,
    use_augmented_data: bool,
    use_augmented_raw: bool,
    use_raw: bool,
    output_dir: Path | None,
    skip_baseline_assert: bool,
) -> dict[str, Any]:
    """Calibrate on val, evaluate on test.

    All user-facing numeric outputs use **Fineness** (same scale as label CSV: model uses
    label/100; we multiply by ``FINENESS_SCALE`` for reports and plots). Normalized tensors
    are kept only under ``internal_normalized_0_1`` for debugging.
    """

    root = project_root()
    run_dir = resolve_run_dir(run_dir)
    ckpt = resolve_checkpoint(run_dir, checkpoint)
    config = load_training_config(run_dir)
    if not skip_baseline_assert:
        assert_convnext_baseline(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = build_convnext_inference().to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    val_loader = make_eval_loader(
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmented_data=use_augmented_data,
        use_augmented_raw=use_augmented_raw,
        use_raw=use_raw,
    )
    test_loader = make_eval_loader(
        "test",
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmented_data=use_augmented_data,
        use_augmented_raw=use_augmented_raw,
        use_raw=use_raw,
    )

    y_val, p_val = collect_predictions(model, val_loader, device)
    y_test, p_test = collect_predictions(model, test_loader, device)

    cp = SplitConformalRegressor(alpha=alpha)
    half_width_norm = cp.calibrate(y_val, p_val)
    lo, hi = cp.predict_interval(p_test)

    scale = FINENESS_SCALE
    y_test_f = y_test * scale
    p_test_f = p_test * scale
    lo_f = lo * scale
    hi_f = hi * scale
    half_width_f = half_width_norm * scale

    baseline_f = point_metrics(y_test_f, p_test_f)
    conf_f = conformal_metrics(y_test_f, lo_f, hi_f)

    metrics_fineness: dict[str, Any] = {
        "baseline_point": baseline_f,
        "conformal_interval": conf_f,
        "nominal_coverage": 1.0 - alpha,
        "alpha": alpha,
        "calibration_half_width": half_width_f,
        "n_calib": int(y_val.size),
        "n_test": int(y_test.size),
    }

    metrics_norm = {
        "baseline_point": point_metrics(y_test, p_test),
        "conformal_interval": conformal_metrics(y_test, lo, hi),
        "nominal_coverage": 1.0 - alpha,
        "alpha": alpha,
        "calibration_half_width": half_width_norm,
        "n_calib": int(y_val.size),
        "n_test": int(y_test.size),
    }

    sweep_alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    alpha_sweep: list[dict[str, float]] = []
    for a in sweep_alphas:
        cpa = SplitConformalRegressor(alpha=a)
        cpa.calibrate(y_val, p_val)
        lo_a, hi_a = cpa.predict_interval(p_test)
        m = conformal_metrics(y_test, lo_a, hi_a)
        alpha_sweep.append(
            {
                "alpha": a,
                "nominal_coverage": 1.0 - a,
                "empirical_coverage": m["coverage"],
                "mean_interval_width": m["mean_width"] * scale,
            }
        )

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "training_config": config,
        "split_conformal": {
            "method": "symmetric_absolute_residual_split_conformal",
            "calibration_split": "val",
            "evaluation_split": "test",
            "leakage_note": (
                "Calibration scores use the validation split (no gradients on val during "
                "training). For strict independence from model selection, reserve a separate "
                "calibration fold before training."
            ),
        },
        "metrics": metrics_fineness,
        "alpha_sweep": alpha_sweep,
        "internal_normalized_0_1": metrics_norm,
    }

    logger.info(
        "Conformal (Fineness): MAE=%.4f coverage=%.3f (nominal %.3f) mean_width=%.4f",
        baseline_f["mae"],
        conf_f["coverage"],
        1.0 - alpha,
        conf_f["mean_width"],
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = (root / out_path).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        out["conformal_dir"] = str(out_path)
        logger.info("Conformal output directory %s", out_path)

    out["_arrays_for_plot"] = {
        "y_test": y_test,
        "p_test": p_test,
        "lo": lo,
        "hi": hi,
    }
    return out
