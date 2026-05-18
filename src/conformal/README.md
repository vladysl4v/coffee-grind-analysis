# Conformal regression

This module adds **split conformal prediction** on top of a trained point regressor (ConvNeXt-Small baseline). It does **not** change training; it loads frozen weights and builds statistically motivated **prediction intervals** on held-out data.

## Idea in plain language

1. You already have a model that outputs one number per image (predicted fineness on the same normalized scale the dataloader uses: CSV value divided by 100).

2. On a **calibration** set we measure how far off the model was: absolute errors `|y − ŷ|`.

3. Take a high quantile of those errors (with a small finite-sample correction). Call that half-width `q`.

4. For a **new** image, the interval `[ŷ − q, ŷ + q]` is designed so that, under ideal assumptions (exchangeable data, same distribution as calibration), the true label falls inside the interval at least a `(1 − α)` fraction of the time in the long run.

So conformal adds **uncertainty bands** around the same point prediction, with a clear **nominal** coverage target tied to **α**.

## Why split conformal here

- The dataset is modest-sized; we want a **distribution-free** guarantee without retraining auxiliary quantile networks (**CQR** would be heavier).
- We keep **training untouched** and only consume `config.json` + `models/epoch_*.pt`.
- **Calibration** uses the **validation** split; **coverage and width** are reported on **test**. (See `leakage_note` in the JSON report: val was used for training monitoring; for strict independence from any tuning, reserve a separate calibration fold before fitting the model.)

## File layout

| File | Role |
|------|------|
| `constants.py` | `FINENESS_SCALE` (100) — maps model outputs to **Fineness** units used in label CSVs. |
| `split.py` | Finite-sample quantile of absolute residuals; `SplitConformalRegressor` (calibrate → predict interval). |
| `metrics.py` | MAE / RMSE for points; empirical **coverage** and **mean/median width** for intervals. |
| `pipeline.py` | Resolve run directory and checkpoint, load ConvNeXt-Small, run val→calibration, test→metrics, write `conformal_report.json` (Fineness-first; normalized copy under `internal_normalized_0_1`). |
| `visualize.py` | PNG figures: interval bars, pred vs truth, coverage–width sweep, width histogram. |
| `../run_conformal.py` | CLI entry point. |

## How to run

From the project root (same as other scripts):

```bash
# Default: picks run_001, run001, or run_002 if present under convnext_small, else latest run_*
uv run python src/run_conformal.py --alpha 0.10

# Explicit run (recommended after renaming your training folder)
uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_001 --alpha 0.10

# Pick a specific checkpoint and quieter logs
uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_001 --checkpoint data/runs/convnext_small/run_001/models/epoch_030.pt --alpha 0.10 --quiet

# Same data switches as training (if you trained on augmented/raw paths)
uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_001 --augmented-data
```

Outputs are written to **``<run_dir>/conformal/``** (e.g. ``data/runs/convnext_small/run_005/conformal/``). After training, ``train.py`` runs this step automatically unless ``--skip-conformal``.

- `conformal_report.json` — primary numbers are **Fineness** (0–100 style values as in CSVs).
- `conformal_test_errorbars.png` — first test samples: interval hits/misses in colour.
- `conformal_pred_vs_truth.png` — scatter of truth vs point prediction; colour = interval hit/miss.
- `conformal_coverage_width.png` — empirical vs nominal coverage sweep; mean width; **horizontal line at target `1 − α`** for the α you passed.
- `conformal_width_histogram.png` — distribution of interval widths on the test set.

## Metrics (Fineness units)

All of the following refer to the **same Fineness scale as `train.csv` / `val.csv` / `test.csv`** (model internally uses ÷100; reports multiply back).

| Term | Meaning |
|------|--------|
| **MAE / RMSE** | Point prediction error; conformal does **not** change the point model, so baseline MAE is the same with or without intervals. |
| **Coverage** | Fraction of test points whose **true** fineness lies inside `[lower, upper]`. Compare to **nominal** `1 − α` (e.g. α = 0.1 → 90% target). With small calibration `n`, empirical coverage often sits **at or above** nominal (conservative). |
| **Mean / median width** | Average size of the interval in Fineness units — **efficiency**: narrower is better *if* coverage stays near nominal. |

The JSON field `internal_normalized_0_1` is optional for debugging only; prefer **`metrics`** for tables and papers.
