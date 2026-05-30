"""Split Conformal Prediction for coffee grind fineness regression.

This package is independent of ``train.py`` and only consumes frozen weights
plus CSV/image paths from ``data_loader``.
"""

from .constants import FINENESS_SCALE
from .metrics import conformal_metrics, point_metrics
from .split import SplitConformalRegressor, finite_sample_residual_quantile

__all__ = [
    "FINENESS_SCALE",
    "SplitConformalRegressor",
    "finite_sample_residual_quantile",
    "conformal_metrics",
    "point_metrics",
]
