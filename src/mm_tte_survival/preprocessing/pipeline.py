"""Split-aware preprocessing: train-only imputation + normalization.

Provenance of feature groups (clinical / cytogenetic / omics / programs) is
preserved so downstream layers know which features are binary cytogenetic flags
vs continuous labs/PCs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ProcessedCohort:
    df: pd.DataFrame
    feature_cols: list[str]
    groups: dict
    train_mask: pd.Series
    time_col: str
    event_col: str

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols)


class PreprocessingPipeline:
    """fit() learns imputers/scalers on TRAIN ONLY; transform() applies them."""

    def __init__(self, normalization: str = "standardize", imputation: str = "median",
                 winsorize: float | None = None, binary_cols: list[str] | None = None):
        self.normalization = normalization
        self.imputation = imputation
        self.winsorize = winsorize
        self.binary_cols = set(binary_cols or [])
        self._impute = None
        self._mu = None
        self._sd = None
        self._cols = None

    def fit(self, train_df: pd.DataFrame, cols: list[str]) -> "PreprocessingPipeline":
        self._cols = list(cols)
        X = train_df[cols].apply(pd.to_numeric, errors="coerce")
        self._impute = X.median() if self.imputation == "median" else X.mean()
        Xf = X.fillna(self._impute).fillna(0.0)
        self._mu = Xf.mean()
        self._sd = Xf.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self._cols].apply(pd.to_numeric, errors="coerce").fillna(self._impute).fillna(0.0)
        if self.winsorize:
            lo, hi = X.quantile(self.winsorize), X.quantile(1 - self.winsorize)
            X = X.clip(lower=lo, upper=hi, axis=1)
        if self.normalization == "standardize":
            scaled = (X - self._mu) / self._sd
            # leave binary cytogenetic flags un-scaled
            for c in self.binary_cols:
                if c in X.columns:
                    scaled[c] = X[c]
            return scaled
        return X

    def fit_transform(self, cohort_df: pd.DataFrame, cols: list[str],
                      train_mask: pd.Series) -> pd.DataFrame:
        self.fit(cohort_df[train_mask.values], cols)
        return self.transform(cohort_df)


def build_preprocessor(normalization="standardize", imputation="median",
                       feature_selection=None, fit_on="train_only", binary_cols=None):
    assert fit_on == "train_only", "preprocessing must be fit on train only"
    return PreprocessingPipeline(normalization=normalization, imputation=imputation,
                                 binary_cols=binary_cols)
