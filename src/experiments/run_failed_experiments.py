"""
Rerun experiments that failed during the main run.

Reads failed_runs.log and reruns every unique command logged as FAILED.
RERUN FAILED entries are skipped to avoid re-running things that already
failed on a previous rerun pass.

Commands that don't already specify --batch-size are automatically run
with --batch-size 16 to avoid the OOM errors that caused the original failure.

Usage
-----
uv run python run_failed_experiments.py
uv run python run_failed_experiments.py --dry-run
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

_ROOT       = Path(__file__).parent
_FAILED_LOG = _ROOT / "failed_runs.log"


def parse_failed_commands(log_path: Path) -> list[str]:
    """Return unique commands from FAILED (not RERUN FAILED) log entries."""
    if not log_path.exists():
        return []

    commands: list[str] = []
    seen: set[str] = set()
    lines = log_path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if "] FAILED " in line and "RERUN FAILED" not in line:
            for j in range(i + 1, len(lines)):
                cmd = lines[j].strip()
                if cmd:
                    if "--batch-size" not in cmd:
                        cmd = cmd + " --batch-size 16"
                    if cmd not in seen:
                        seen.add(cmd)
                        commands.append(cmd)
                    break
        i += 1

    return commands


def log_failure(cmd: str, returncode: int) -> None:
    with open(_FAILED_LOG, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] RERUN FAILED (exit {returncode})\n  {cmd}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rerun experiments logged as FAILED in failed_runs.log.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running them")
    args = parser.parse_args()

    commands = parse_failed_commands(_FAILED_LOG)

    if not commands:
        print(f"No FAILED entries found in {_FAILED_LOG.name}.")
        return

    total = len(commands)
    print(f"Found {total} failed experiment(s) to rerun.")
    if args.dry_run:
        print("DRY RUN — no training will happen\n")

    failed = 0
    for i, cmd in enumerate(commands):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cmd}\n")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, shell=True, cwd=_ROOT)

        if result.returncode != 0:
            failed += 1
            log_failure(cmd, result.returncode)
            print(f"  !! Failed (exit {result.returncode}) — logged to {_FAILED_LOG.name}, continuing")

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Done. {total} runs attempted, {failed} failed.")
        if failed:
            print(f"Failed runs logged to: {_FAILED_LOG}")


if __name__ == "__main__":
    main()
