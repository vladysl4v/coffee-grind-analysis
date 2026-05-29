"""
Run all training experiments from TRAINING_PLAN.md in memory-safe order.

Catches OOM and other failures gracefully, logs them to failed_runs.log,
and continues with the next experiment.

Usage
-----
uv run python run_all_experiments.py
uv run python run_all_experiments.py --dry-run        # print commands without running
uv run python run_all_experiments.py --start-at 12   # resume from experiment 12
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

_ROOT       = Path(__file__).parent
_FAILED_LOG = _ROOT / "failed_runs.log"

BASE = "uv run python src/train.py --epochs 100 --adamw --cosine-lr --huber --huber-delta 0.03 --lr 3e-4"
ADV  = "--adversarial --adv-epsilon 0.02 --adv-weight 0.4"
AUG  = "--online-augment"


def _variants(model, unfreeze=None, batch_size=None):
    """Return 4 aug/adv variant commands for a model config."""
    opts = f"--model {model}"
    if unfreeze:
        opts += f" --unfreeze-after {unfreeze}"
    if batch_size:
        opts += f" --batch-size {batch_size}"
    base = f"{BASE} {opts}"
    return [
        base,
        f"{base} {AUG}",
        f"{base} {ADV}",
        f"{base} {AUG} {ADV}",
    ]


# Ordered safest/smallest → largest/most likely to OOM
EXPERIMENTS = [
    *_variants("fpn_resnet",        batch_size=16)
]


def log_failure(cmd: str, returncode: int) -> None:
    with open(_FAILED_LOG, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FAILED (exit {returncode})\n  {cmd}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running them")
    parser.add_argument("--start-at", type=int, default=0, metavar="N",
                        help="skip the first N experiments (resume after partial run)")
    args = parser.parse_args()

    total = len(EXPERIMENTS)
    print(f"Total experiments: {total}")
    if args.dry_run:
        print("DRY RUN — no training will happen\n")

    failed = 0
    for i, cmd in enumerate(EXPERIMENTS):
        if i < args.start_at:
            print(f"[{i+1}/{total}] Skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cmd}\n")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, shell=True, cwd=_ROOT)

        if result.returncode != 0:
            failed += 1
            log_failure(cmd, result.returncode)
            print(f"  !! Failed (exit {result.returncode}) — logged to {_FAILED_LOG.name}, continuing")

    print(f"\n{'='*60}")
    if not args.dry_run:
        print(f"Done. {total - args.start_at} runs attempted, {failed} failed.")
        if failed:
            print(f"Failed runs logged to: {_FAILED_LOG}")


if __name__ == "__main__":
    main()
