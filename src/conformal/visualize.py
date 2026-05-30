"""Figures for split conformal regression (Fineness scale in titles; axes stay minimal)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_test_interval_errorbars(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fineness_scale: float,
    out_path: Path,
    max_samples: int = 24,
    title: str = "Test intervals — Fineness 0–100 (label units)",
) -> None:
    """Horizontal intervals: green if truth lies inside, red otherwise."""

    y_true = np.asarray(y_true, dtype=np.float64).ravel() * fineness_scale
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel() * fineness_scale
    lower = np.asarray(lower, dtype=np.float64).ravel() * fineness_scale
    upper = np.asarray(upper, dtype=np.float64).ravel() * fineness_scale

    n = min(max_samples, len(y_true))
    idx = np.arange(n)
    covered = (y_true[:n] >= lower[:n]) & (y_true[:n] <= upper[:n])

    fig, ax = plt.subplots(figsize=(10, max(4.2, 0.38 * n)))
    for i in idx:
        col = "#2ca02c" if covered[i] else "#d62728"
        ax.plot([lower[i], upper[i]], [i, i], color=col, linewidth=2.4, alpha=0.9, solid_capstyle="round")
        ax.scatter(
            y_true[i],
            i,
            color="#1f4f1f" if covered[i] else "#7f0000",
            s=42,
            zorder=4,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            y_pred[i],
            i,
            color="#1f77b4" if covered[i] else "#ff7f0e",
            s=30,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # Build legend without duplicate dummy labels
    h1 = ax.scatter([], [], c="#1f4f1f", s=42, edgecolors="white", linewidths=0.6, label="Truth — covered")
    h2 = ax.scatter([], [], c="#7f0000", s=42, edgecolors="white", linewidths=0.6, label="Truth — not covered")
    h3 = ax.scatter([], [], c="#1f77b4", s=30, edgecolors="white", linewidths=0.5, label="Prediction — covered")
    h4 = ax.scatter([], [], c="#ff7f0e", s=30, edgecolors="white", linewidths=0.5, label="Prediction — not covered")
    h5 = ax.plot([], [], color="#2ca02c", linewidth=2.4, label="Interval (covers truth)")[0]
    h6 = ax.plot([], [], color="#d62728", linewidth=2.4, label="Interval (misses truth)")[0]
    ax.legend(handles=[h1, h2, h3, h4, h5, h6], loc="lower right", fontsize=9, framealpha=0.92)

    ax.set_yticks(idx)
    ax.set_yticklabels([f"#{i}" for i in idx])
    ax.set_xlabel("Fineness")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_coverage_width_tradeoff(
    sweep_rows: list[dict[str, float]],
    *,
    out_path: Path,
    reference_alpha: float | None = None,
    title: str = "Coverage vs interval width (Fineness 0–100)",
) -> None:
    """Nominal vs empirical coverage; mean width on twin axis; reference line at 1−α."""

    nom = np.array([row["nominal_coverage"] for row in sweep_rows], dtype=float)
    emp = np.array([row["empirical_coverage"] for row in sweep_rows], dtype=float)
    widths = np.array([row["mean_interval_width"] for row in sweep_rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.plot(nom, emp, "o-", color="tab:blue", linewidth=2, markersize=7, label="Empirical coverage")
    ax1.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.2, label="Perfect calibration")

    if reference_alpha is not None and 0.0 < reference_alpha < 1.0:
        target_cov = 1.0 - reference_alpha
        ax1.axhline(
            y=target_cov,
            color="tab:green",
            linestyle=":",
            linewidth=2,
            label=f"Target coverage 1−α = {target_cov:.0%} (α = {reference_alpha:g})",
        )

    ax1.set_xlabel("Nominal coverage")
    ax1.set_ylabel("Empirical coverage (test)")
    ax1.set_xlim(float(nom.min()) - 0.02, 1.0)
    lo = min(float(emp.min()), float(nom.min())) - 0.03
    ax1.set_ylim(max(0.55, lo), 1.02)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(nom, widths, "s--", color="tab:red", alpha=0.75, markersize=6, linewidth=1.5, label="Mean interval width")
    ax2.set_ylabel("Mean width")

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=8, framealpha=0.95)
    ax1.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_interval_width_histogram(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fineness_scale: float,
    out_path: Path,
    title: str = "Test interval widths — Fineness 0–100",
    bins: int = 16,
) -> None:
    """Histogram of (upper − lower) on the test set at the chosen calibration."""

    lower = np.asarray(lower, dtype=np.float64).ravel() * fineness_scale
    upper = np.asarray(upper, dtype=np.float64).ravel() * fineness_scale
    widths = upper - lower

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(widths, bins=bins, color="steelblue", edgecolor="white", alpha=0.88)
    ax.axvline(float(np.mean(widths)), color="tab:orange", linestyle="--", linewidth=2, label=f"Mean = {np.mean(widths):.2f}")
    ax.axvline(float(np.median(widths)), color="tab:green", linestyle=":", linewidth=2, label=f"Median = {np.median(widths):.2f}")
    ax.set_xlabel("Fineness")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_pred_vs_truth_with_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fineness_scale: float,
    out_path: Path,
    title: str = "Predictions vs truth — Fineness 0–100",
) -> None:
    """Scatter truth vs point prediction; colour encodes interval hit/miss."""

    y_true = np.asarray(y_true, dtype=np.float64).ravel() * fineness_scale
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel() * fineness_scale
    lower = np.asarray(lower, dtype=np.float64).ravel() * fineness_scale
    upper = np.asarray(upper, dtype=np.float64).ravel() * fineness_scale

    covered = (y_true >= lower) & (y_true <= upper)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(
        y_true[covered],
        y_pred[covered],
        s=40,
        c="#2ca02c",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.5,
        label="Truth inside interval",
    )
    ax.scatter(
        y_true[~covered],
        y_pred[~covered],
        s=52,
        c="#d62728",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.5,
        label="Truth outside interval",
    )
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.45, linewidth=1.1, label="Ideal y = ŷ")
    ax.set_xlabel("Fineness")
    ax.set_ylabel("Fineness")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
