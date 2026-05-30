"""
Evaluate split conformal regression for a saved training run.

Reports and figures are written to ``<run_dir>/conformal/`` by default.

Example
-------
uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_005
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from conformal.evaluate import evaluate_and_save, resolve_eval_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split conformal regression on val→test.")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory (e.g. data/runs/convnext_small/run_005)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Weights path (default: models/best.pt, else latest epoch_*.pt)",
    )
    p.add_argument("--alpha", type=float, default=0.10, help="Target miscoverage α (default 0.1 → 90%% intervals)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <run-dir>/conformal/)",
    )
    p.add_argument(
        "--skip-baseline-assert",
        action="store_true",
        help="Do not require convnext_small + adversarial + unfreeze flags in config.json",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logs on stderr")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    run_dir = Path(args.run_dir)
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = resolve_eval_checkpoint(run_dir)

    out_dir = evaluate_and_save(
        run_dir,
        checkpoint=ckpt,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        skip_baseline_assert=args.skip_baseline_assert,
    )

    report = json.loads((out_dir / "conformal_report.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
