"""
Evaluation metrics.

Spotter does not publish the metric they score with, so nothing here optimises
for a single one. Every model is reported on absolute-error and
percentage-error families side by side:

  RMSE   punishes large misses; dominated by long, expensive loads
  MAE    even weighting in dollars
  MAPE   percentage error, so a $50 miss on a $400 load matters as much as a
         $600 miss on a $5,000 one - short hauls drive this
  MedAPE median percentage error, largely immune to the corrupt labels

A model that wins on RMSE can lose badly on MAPE. Reporting both keeps the
choice honest rather than tuned to a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Return the full metric family for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    ape = np.abs(error) / y_true

    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(ape) * 100),
        "MedAPE": float(np.median(ape) * 100),
        "R2": float(1 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "Bias": float(np.mean(error)),
    }


def scoreboard(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Tabulate {model_name: metrics} sorted by MAE."""
    frame = pd.DataFrame(results).T
    return frame.sort_values("MAE")[["RMSE", "MAE", "MAPE", "MedAPE", "R2", "Bias"]]


def smearing_factor(residuals_log) -> float:
    """
    Duan's smearing estimate.

    Fitting on log(rate) and exponentiating predicts the *median*, not the mean,
    which biases predictions low. Multiplying by mean(exp(residual)) corrects it.
    Worth doing explicitly rather than leaving a systematic under-prediction in
    place - the Bias column shows whether it worked.
    """
    return float(np.mean(np.exp(np.asarray(residuals_log, dtype=float))))
