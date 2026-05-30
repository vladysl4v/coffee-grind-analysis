"""Finite-sample split conformal regression (absolute residual scores)."""

from __future__ import annotations

import numpy as np


def finite_sample_residual_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Half-width for symmetric intervals at miscoverage ``alpha``.

    Uses the standard finite-sample correction (order statistic on
    ``|y - \\hat y|`` with an effective sample size ``n + 1``), so that under
    exchangeability and correct specification the marginal coverage is at least
    ``1 - alpha`` for two-sided symmetric intervals.

    Parameters
    ----------
    abs_residuals
        Non-negative calibration scores ``|y_i - f(x_i)|``.
    alpha
        Target miscoverage in ``(0, 1)`` (e.g. ``0.1`` for 90% nominal).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    scores = np.asarray(abs_residuals, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("abs_residuals must be non-empty")
    sorted_scores = np.sort(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(sorted_scores[k - 1])


class SplitConformalRegressor:
    """Symmetric residual split conformal layer on top of a point predictor."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.half_width_: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fit ``half_width_`` from calibration residuals (same scale as ``y``)."""

        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        scores = np.abs(y_true - y_pred)
        self.half_width_ = finite_sample_residual_quantile(scores, self.alpha)
        return self.half_width_

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.half_width_ is None:
            raise RuntimeError("Call calibrate() before predict_interval()")
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        q = self.half_width_
        return y_pred - q, y_pred + q
