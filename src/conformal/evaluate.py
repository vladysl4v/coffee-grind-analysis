"""High-level conformal evaluation + figures for a training run directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from conformal.constants import FINENESS_SCALE
from conformal.pipeline import (
    load_training_config,
    resolve_checkpoint,
    resolve_run_dir,
    run_split_conformal_eval,
)
from conformal.visualize import (
    plot_coverage_width_tradeoff,
    plot_interval_width_histogram,
    plot_pred_vs_truth_with_intervals,
    plot_test_interval_errorbars,
)

logger = logging.getLogger(__name__)


def conformal_output_dir(run_dir: Path) -> Path:
    return Path(run_dir).resolve() / "conformal"


def evaluate_and_save(
    run_dir: Path | str,
    *,
    checkpoint: Path | None = None,
    alpha: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: Path | None = None,
    skip_baseline_assert: bool = False,
) -> Path:
    """Run split conformal on val→test and write JSON + PNG under ``run_XXX/conformal/``."""

    run_dir = resolve_run_dir(Path(run_dir))
    out_dir = Path(output_dir).resolve() if output_dir is not None else conformal_output_dir(run_dir)
    config = load_training_config(run_dir)

    out = run_split_conformal_eval(
        run_dir=run_dir,
        checkpoint=checkpoint,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmented_data=bool(config.get("augmented_data")),
        use_augmented_raw=bool(config.get("augmented_raw_precomputed")),
        use_raw=bool(config.get("raw")),
        output_dir=out_dir,
        skip_baseline_assert=skip_baseline_assert,
    )

    ar = out.pop("_arrays_for_plot", None)
    sweep = out.get("alpha_sweep", [])
    scale = FINENESS_SCALE

    if ar is not None:
        plot_test_interval_errorbars(
            ar["y_test"],
            ar["p_test"],
            ar["lo"],
            ar["hi"],
            fineness_scale=scale,
            out_path=out_dir / "conformal_test_errorbars.png",
        )
        plot_pred_vs_truth_with_intervals(
            ar["y_test"],
            ar["p_test"],
            ar["lo"],
            ar["hi"],
            fineness_scale=scale,
            out_path=out_dir / "conformal_pred_vs_truth.png",
        )
        plot_interval_width_histogram(
            ar["lo"],
            ar["hi"],
            fineness_scale=scale,
            out_path=out_dir / "conformal_width_histogram.png",
        )
    if sweep:
        plot_coverage_width_tradeoff(
            sweep,
            out_path=out_dir / "conformal_coverage_width.png",
            reference_alpha=alpha,
        )

    report_path = out_dir / "conformal_report.json"
    report_path.write_text(
        json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2),
        encoding="utf-8",
    )
    logger.info("Conformal artifacts saved to %s", out_dir)
    return out_dir


def resolve_eval_checkpoint(run_dir: Path, *, prefer_best: bool = True) -> Path:
    """Pick ``models/best.pt``, else latest ``epoch_*.pt``."""

    run_dir = Path(run_dir).resolve()
    best = run_dir / "models" / "best.pt"
    if prefer_best and best.is_file():
        return best
    return resolve_checkpoint(run_dir, None)
