"""
Evaluate split conformal regression for a saved ConvNeXt-Small run.

Example
-------
uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_001 --alpha 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from conformal.constants import FINENESS_SCALE
from conformal.pipeline import run_split_conformal_eval
from conformal.visualize import (
    plot_coverage_width_tradeoff,
    plot_interval_width_histogram,
    plot_pred_vs_truth_with_intervals,
    plot_test_interval_errorbars,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split conformal regression on val→test.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Training run directory (default: first existing among run_001, run001, run_002 "
            "under data/runs/convnext_small, else latest run_*)"
        ),
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Weights path (default: latest epoch_*.pt)")
    p.add_argument("--alpha", type=float, default=0.10, help="Target miscoverage α (default 0.1 → 90%% intervals)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("data/runs/conformal_eval"), help="JSON + figures")
    p.add_argument("--augmented-data", action="store_true")
    p.add_argument("--augmented-raw-precomputed", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument(
        "--skip-baseline-assert",
        action="store_true",
        help="Do not require convnext_small + adversarial + unfreeze flags in config.json",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs (JSON still printed to stdout)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    out = run_split_conformal_eval(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmented_data=args.augmented_data,
        use_augmented_raw=args.augmented_raw_precomputed,
        use_raw=args.raw,
        output_dir=args.output_dir,
        skip_baseline_assert=args.skip_baseline_assert,
    )

    ar = out.pop("_arrays_for_plot", None)
    sweep = out.get("alpha_sweep", [])
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            reference_alpha=args.alpha,
        )

    print(json.dumps({k: v for k, v in out.items() if k != "_arrays_for_plot"}, indent=2))
    logger.info("Figures and report under %s", out_dir.resolve())


if __name__ == "__main__":
    main()
