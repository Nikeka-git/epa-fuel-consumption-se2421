"""Deterministic text sanitation and training-fitted Final feature extraction.

Original source columns remain unchanged. Sanitation rules precede experiments
and never inspect targets. Empty text folds contribute an all-zero sparse block.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, MIDTERM_FEATURES, NUMERIC_FEATURES, select_features, validate_feature_config

SANITIZER_VERSION = "1.0"
ARMS = ("structured", "structured_model", "structured_engine", "structured_model_engine")
TEXT_FIELDS = ("model_name", "engine_description")
ARM_TEXT = {"structured": (), "structured_model": ("model_name",), "structured_engine": ("engine_description",), "structured_model_engine": TEXT_FIELDS}
NUMBER = r"[-+]?\d+(?:[.,]\d+)*"
LEAKAGE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:gas\s+)?guzzler\b",
    rf"\b(?:fuelCost[A-Z0-9]*|co2[A-Z0-9]*|barrels[A-Z0-9]*|feScore[A-Z0-9]*|ghgScore[A-Z0-9]*|smartwayScore|youSaveSpend)\s*[:=]?\s*[$€£]?\s*{NUMBER}(?:\s*(?:g\s*/\s*(?:km|mi)|kg\s*/\s*year|usd|dollars?))?\b",
    rf"\b{NUMBER}\s*(?:mpge?|miles\s+per\s+gallon)\b",
    rf"\b(?:mpge?|miles\s+per\s+gallon)\s*[:=]?\s*{NUMBER}\b",
    rf"\b{NUMBER}\s*(?:l(?:it(?:er|re)s?)?\s*/\s*100\s*km|l\s+per\s+100\s*km)\b",
    rf"\b(?:l\s*/\s*100\s*km)\s*[:=]?\s*{NUMBER}\b",
    rf"\b(?:fuel\s*(?:economy|consumption)|(?:epa\s*)?(?:combined|city|highway)\s*(?:mpge?|fuel\s*economy|consumption)?)\s*[:=]?\s*{NUMBER}(?:\s*(?:mpge?|l\s*/\s*100\s*km))?\b",
    rf"\b(?:annual\s*)?fuel\s*cost(?:s|\d)?\s*[:=]?\s*[$€£]?\s*{NUMBER}(?:\s*(?:usd|dollars?))?\b",
    rf"\b(?:co2(?:TailpipeGpm)?|emissions?(?:\s*score)?|ghg\s*score|ghgScore[A-Z]?|feScore|efficiency\s*score|smartway\s*score|smartwayScore|youSaveSpend|barrels\d?)\s*[:=]?\s*[$€£]?\s*{NUMBER}(?:\s*(?:g\s*/\s*(?:km|mi)|kg\s*/\s*year))?\b",
    rf"\b{NUMBER}\s*(?:g\s*/\s*(?:km|mi)|kg\s*(?:co2\s*)?/\s*year)\b",
    rf"[$€£]\s*{NUMBER}(?:,\d{{3}})*(?:\s*(?:/\s*year|per\s*year))?",
    r"\b(?:feScore|ghgScore[A-Z]?|smartwayScore|youSaveSpend|fuelCost[A-Z0-9]*|co2TailpipeGpm[A-Z]*|barrels[A-Z0-9]*)\b",
    r"\b(?:mpge?|fuel\s*(?:economy|consumption|costs?)|co2|emissions?|efficiency\s*score|ghg\s*score|smartway(?:\s*score)?)\b",
))


def sanitize_text(value) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if not isinstance(value, str):
        raise ValueError("Text predictors must be strings or missing")
    text = value
    for pattern in LEAKAGE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class SafeTextVectorizer(BaseEstimator, TransformerMixin):
    """Sanitize before fitting TF-IDF and handle entirely empty training folds."""
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=1, sublinear_tf=True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

    @staticmethod
    def _texts(X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("Text block requires one source column")
            X = X.iloc[:, 0]
        return [sanitize_text(value) for value in X]

    def fit(self, X, y=None):
        texts = self._texts(X)
        vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=tuple(self.ngram_range), min_df=self.min_df, sublinear_tf=self.sublinear_tf)
        try:
            vectorizer.fit(texts)
        except ValueError as exc:
            if not any(message in str(exc) for message in ("empty vocabulary", "After pruning, no terms remain")):
                raise
            self.vectorizer_ = None
        else:
            self.vectorizer_ = vectorizer
        self.empty_training_vocabulary_ = self.vectorizer_ is None
        return self

    def transform(self, X):
        texts = self._texts(X)
        if not hasattr(self, "vectorizer_"):
            raise ValueError("Text vectorizer is not fitted")
        if self.vectorizer_ is None:
            return sparse.csr_matrix((len(texts), 1), dtype=float)
        return self.vectorizer_.transform(texts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["empty_training_text_block"], dtype=object) if self.vectorizer_ is None else self.vectorizer_.get_feature_names_out()


class FinalFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, arm="structured"):
        self.arm = arm

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if self.arm not in ARM_TEXT:
            raise ValueError("Unknown Final feature arm")
        out = select_features(X)
        for field in ARM_TEXT[self.arm]:
            if field not in X:
                raise ValueError(f"Missing preserved Final text column: {field}")
            out[field] = X[field].map(lambda value: "" if pd.isna(value) else value)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MIDTERM_FEATURES + ARM_TEXT[self.arm], dtype=object)


def validate_stage_config(stage: dict) -> None:
    if stage.get("stage") != "final" or tuple(stage.get("arms", ())) != ARMS:
        raise ValueError("Final comparison requires the four predefined arms")
    if stage.get("estimator") != "Ridge" or stage.get("solver") != "lsqr":
        raise ValueError("Final comparator must use Ridge with lsqr")
    alpha = stage.get("alpha_grid", [])
    if not alpha or len(set(alpha)) != len(alpha) or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not np.isfinite(a) or a <= 0 for a in alpha):
        raise ValueError("alpha_grid must contain distinct positive finite numbers")
    if stage["text"].get("sanitizer_version") != SANITIZER_VERSION:
        raise ValueError("Sanitizer version differs from this implementation")
    if stage["text"].get("max_features") != 5000 or stage["text"].get("ngram_range") != [1, 2]:
        raise ValueError("Predefined TF-IDF budget is max_features=5000, ngrams=(1,2)")


def build_text_pipeline(config: dict, stage: dict, arm: str, alpha: float) -> Pipeline:
    validate_feature_config(config)
    validate_stage_config(stage)
    if arm not in ARMS or alpha not in stage["alpha_grid"]:
        raise ValueError("Arm/alpha is outside the declared training-CV search")
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    blocks = [("numeric", numeric, list(NUMERIC_FEATURES)), ("categorical", categorical, list(CATEGORICAL_FEATURES))]
    for field in ARM_TEXT[arm]:
        settings = stage["text"]
        blocks.append((field, SafeTextVectorizer(settings["max_features"], tuple(settings["ngram_range"]), settings.get("min_df", 1), settings.get("sublinear_tf", True)), field))
    return Pipeline([("select", FinalFeatureSelector(arm)),
                     ("preprocess", ColumnTransformer(blocks, remainder="drop", sparse_threshold=1.0)),
                     ("model", Ridge(alpha=alpha, solver=stage["solver"]))])
