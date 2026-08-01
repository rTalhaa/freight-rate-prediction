"""
Rolling-origin cross-validation.

A single Sep-Oct holdout is one draw. Whether the hybrid genuinely beats ridge
by $2 or got a lucky two months is not something one split can answer, so model
selection runs over several folds that each reproduce the real setup: train on
everything up to a cut-off, predict the two months after it.

Folds always move forward. Nothing here shuffles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import corrupt_label_mask, drop_corrupt_labels
from .features import fit_feature_stats
from .metrics import evaluate


@dataclass
class Fold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(
    frame: pd.DataFrame, horizon_days: int = 61, n_folds: int = 4, min_train_days: int = 120
) -> list[Fold]:
    """
    Expanding-window folds with a fixed forward horizon.

    horizon_days defaults to 61, the length of the real Nov-Dec task, so each
    fold asks the same question the submission does.
    """
    start, end = frame["date"].min(), frame["date"].max()
    folds = []
    for i in range(n_folds):
        test_end = end - pd.Timedelta(days=horizon_days * i)
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        train_end = test_start - pd.Timedelta(days=1)
        if (train_end - start).days < min_train_days:
            break
        folds.append(Fold(
            name=f"{test_start.date()}..{test_end.date()}",
            train_end=train_end, test_start=test_start, test_end=test_end,
        ))
    return list(reversed(folds))


def cross_validate(
    model_factory, frame: pd.DataFrame, folds: list[Fold], clean_test: bool = False
) -> pd.DataFrame:
    """
    Score a model across folds.

    model_factory is a zero-argument callable returning an unfitted model, so
    every fold gets a fresh one.

    Training rows always have corrupt labels removed. Test rows keep theirs
    unless clean_test is set, because the real scoring will not be cleaned.
    Feature statistics are refitted per fold on that fold's training rows only.
    """
    rows = {}
    for fold in folds:
        train = frame.loc[frame["date"] <= fold.train_end]
        test = frame.loc[(frame["date"] >= fold.test_start)
                         & (frame["date"] <= fold.test_end)]
        if train.empty or test.empty:
            continue

        train = drop_corrupt_labels(train)
        if clean_test:
            test = drop_corrupt_labels(test)

        model = model_factory().fit(train, fit_feature_stats(train))
        prediction = model.predict(test)
        actual = test["posted_rate"].to_numpy()

        metrics = evaluate(actual, prediction)
        keep = ~corrupt_label_mask(test).to_numpy()
        metrics["MAE_clean"] = evaluate(actual[keep], prediction[keep])["MAE"]
        metrics["Bias_clean"] = evaluate(actual[keep], prediction[keep])["Bias"]
        metrics["n_train"] = len(train)
        metrics["n_test"] = len(test)
        rows[fold.name] = metrics

    result = pd.DataFrame(rows).T
    result.loc["MEAN"] = result.mean()
    return result


def compare(model_factories: dict, frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Cross-validate several models and return their fold-averaged scores."""
    summary = {}
    for name, factory in model_factories.items():
        scores = cross_validate(factory, frame, folds)
        summary[name] = scores.loc["MEAN"]
    return pd.DataFrame(summary).T.sort_values("MAE")
