"""Strict Midterm feature selection and training-only preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

NUMERIC_FEATURES = ("model_year", "displacement_l", "cylinders")
CATEGORICAL_FEATURES = ("manufacturer", "transmission", "drivetrain", "vehicle_class")
MIDTERM_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_CLASSES = {
    "dummy": DummyRegressor,
    "linear": LinearRegression,
    "knn": KNeighborsRegressor,
    "tree": DecisionTreeRegressor,
}


def validate_feature_config(config: dict) -> None:
    """Refuse accidental additions (including target proxies) to the fixed contract."""
    features = config["features"]
    if tuple(features["numeric"]) != NUMERIC_FEATURES:
        raise ValueError(f"Midterm numeric allowlist must be {NUMERIC_FEATURES}")
    if tuple(features["categorical"]) != CATEGORICAL_FEATURES:
        raise ValueError(f"Midterm categorical allowlist must be {CATEGORICAL_FEATURES}")
    if features.get("text_in_midterm", False):
        raise ValueError("Text features are reserved for Final; Midterm contract is fixed.")


def select_features(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Select only approved X; normalization here is deterministic, never statistical."""
    if config is not None:
        validate_feature_config(config)
    missing = sorted(set(MIDTERM_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictor columns: {missing}")
    selected = frame.loc[:, list(MIDTERM_FEATURES)].copy()
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(selected[name], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError(f"Non-finite nonmissing predictor: {name}")
        selected[name] = values
    for name in CATEGORICAL_FEATURES:
        selected[name] = selected[name].map(
            lambda value: np.nan if pd.isna(value) or not str(value).strip() else str(value).strip()
        ).astype(object)
    return selected


class StrictFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep selection inside saved pipelines so inference uses the same input contract."""

    def fit(self, X: pd.DataFrame, y=None):
        select_features(X)
        self.feature_names_in_ = np.asarray(MIDTERM_FEATURES, dtype=object)
        self.n_features_in_ = len(MIDTERM_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MIDTERM_FEATURES, dtype=object)


def build_preprocessor(config: dict) -> ColumnTransformer:
    validate_feature_config(config)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", verbose_feature_names_out=True)


def build_pipeline(config: dict, model: str) -> Pipeline:
    validate_feature_config(config)
    if model not in MODEL_CLASSES:
        raise ValueError(f"Unknown Midterm model: {model}")
    settings = config["midterm_models"][model]
    estimator_class = MODEL_CLASSES[model]
    if settings["class"] != estimator_class.__name__:
        raise ValueError(f"Unexpected estimator class for {model}: {settings['class']}")
    return Pipeline([
        ("select", StrictFeatureSelector()),
        ("preprocess", build_preprocessor(config)),
        ("model", estimator_class(**settings.get("params", {}))),
    ])


def dense_memory_estimate(frame: pd.DataFrame) -> dict:
    """Upper bound for the float64 encoded training matrix, before allocating it."""
    features = select_features(frame)
    numeric_width = 2 * len(NUMERIC_FEATURES)
    categorical_width = sum(features[name].fillna("Unknown").nunique() for name in CATEGORICAL_FEATURES)
    width = int(numeric_width + categorical_width)
    return {"n_rows": len(features), "upper_bound_columns": width,
            "upper_bound_float64_bytes": int(len(features) * width * 8)}
