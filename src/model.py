"""
Models.

Phase 3 measured the constraint that shapes this file: the task is forward
extrapolation, and tree splits cannot emit a value outside the training range.
LightGBM therefore cannot follow the upward rate trend past the end of the
training window, and lost to plain ridge because of it.

The answer is to split the job. A linear model carries the global structure -
including the trend, which is the part that has to extrapolate - and the trees
fit only what is left over, with no time features at all, so nothing is asked
of them that they cannot do.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import FeatureConfig, FeatureStats, build_features, fit_feature_stats
from .metrics import smearing_factor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbose=-1,
    random_state=0,
)


@dataclass
class Standardiser:
    """Column means and scales, learned once on the training matrix."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardiser":
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mean=X.mean(axis=0), scale=scale)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def _matrix(frame: pd.DataFrame, stats: FeatureStats, config: FeatureConfig,
            expanding: bool, columns=None) -> tuple[np.ndarray, list[str]]:
    features = build_features(frame, stats, config, expanding_lane=expanding)
    if columns is not None:
        features = features[columns]
    return np.nan_to_num(features.to_numpy(dtype=float)), list(features.columns)


class RidgeModel:
    """Ridge on log(rate), with Duan smearing on the way back to dollars."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, config: FeatureConfig | None = None):
        self.alpha = alpha
        self.config = config or FeatureConfig()

    def fit(self, train: pd.DataFrame, feature_stats: FeatureStats | None = None):
        self.feature_stats_ = feature_stats or fit_feature_stats(train)
        X, self.columns_ = _matrix(train, self.feature_stats_, self.config, expanding=True)
        y = np.log(train["posted_rate"].to_numpy())

        self.scaler_ = Standardiser.fit(X)
        self.estimator_ = Ridge(alpha=self.alpha).fit(self.scaler_.apply(X), y)
        self.smearing_ = smearing_factor(y - self.estimator_.predict(self.scaler_.apply(X)))
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        X, _ = _matrix(frame, self.feature_stats_, self.config, expanding=False,
                       columns=self.columns_)
        return self.estimator_.predict(self.scaler_.apply(X))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.exp(self.predict_log(frame)) * self.smearing_, 1.0, None)


class LGBMModel:
    """LightGBM on log(rate). Included as a control, not as the expected winner."""

    name = "lgbm"

    def __init__(self, config: FeatureConfig | None = None, **params):
        self.config = config or FeatureConfig()
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, train: pd.DataFrame, feature_stats: FeatureStats | None = None):
        self.feature_stats_ = feature_stats or fit_feature_stats(train)
        X, self.columns_ = _matrix(train, self.feature_stats_, self.config, expanding=True)
        y = np.log(train["posted_rate"].to_numpy())
        self.estimator_ = lgb.LGBMRegressor(**self.params).fit(X, y)
        self.smearing_ = smearing_factor(y - self.estimator_.predict(X))
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        X, _ = _matrix(frame, self.feature_stats_, self.config, expanding=False,
                       columns=self.columns_)
        return self.estimator_.predict(X)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.exp(self.predict_log(frame)) * self.smearing_, 1.0, None)


class HybridModel:
    """
    Ridge for structure, LightGBM for the residual.

    Stage 1 fits ridge on the full feature set, trend included, so the part of
    the prediction that must extrapolate is linear and does so safely.

    Stage 2 fits LightGBM to the stage-1 residual using features with **no time
    component**. The residual is what linear structure cannot express -
    interactions and non-linear shapes - and none of it needs to extend past the
    training window. The trees are only asked for what they are good at.

    A shrinkage factor damps the stage-2 correction. The residual model is the
    component most likely to have memorised fold-specific noise, so applying it
    at less than full strength trades a little fit for stability.
    """

    name = "hybrid"

    def __init__(self, alpha: float = 1.0, shrinkage: float = 1.0,
                 config: FeatureConfig | None = None, **params):
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.config = config or FeatureConfig()
        # Residual stage sees geography and lane history, never the clock.
        self.residual_config = FeatureConfig(
            time=False,
            geo=self.config.geo,
            lane=self.config.lane,
            interactions=self.config.interactions,
        )
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, train: pd.DataFrame, feature_stats: FeatureStats | None = None):
        self.feature_stats_ = feature_stats or fit_feature_stats(train)
        y = np.log(train["posted_rate"].to_numpy())

        X_linear, self.linear_columns_ = _matrix(
            train, self.feature_stats_, self.config, expanding=True
        )
        self.scaler_ = Standardiser.fit(X_linear)
        self.linear_ = Ridge(alpha=self.alpha).fit(self.scaler_.apply(X_linear), y)
        linear_pred = self.linear_.predict(self.scaler_.apply(X_linear))

        residual = y - linear_pred
        X_residual, self.residual_columns_ = _matrix(
            train, self.feature_stats_, self.residual_config, expanding=True
        )
        self.residual_ = lgb.LGBMRegressor(**self.params).fit(X_residual, residual)

        combined = linear_pred + self.shrinkage * self.residual_.predict(X_residual)
        self.smearing_ = smearing_factor(y - combined)
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        X_linear, _ = _matrix(frame, self.feature_stats_, self.config,
                              expanding=False, columns=self.linear_columns_)
        X_residual, _ = _matrix(frame, self.feature_stats_, self.residual_config,
                                expanding=False, columns=self.residual_columns_)
        return (self.linear_.predict(self.scaler_.apply(X_linear))
                + self.shrinkage * self.residual_.predict(X_residual))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.exp(self.predict_log(frame)) * self.smearing_, 1.0, None)
