"""Shared constants for conformal evaluation (aligned with ``data_loader`` label scaling)."""

from __future__ import annotations

# Labels in CSV are fineness values; the DataLoader divides by this factor for the model.
FINENESS_SCALE: float = 100.0
