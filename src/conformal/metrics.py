"""Point and conformal interval metrics."""

from __future__ import annotations

import numpy as np


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def conformal_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Empirical coverage and mean interval width (same units as ``y_true``)."""

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
    }
