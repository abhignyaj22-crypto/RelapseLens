"""Orthogonalise molecular features against clinical features (train-fit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ClinicalResidualizer:
    """Removes the clinical-explainable component from molecular features so the
    residual carries genuine incremental information."""

    def __init__(self):
        self._lr = None

    def fit(self, Xc_train: np.ndarray, Xm_train: np.ndarray) -> "ClinicalResidualizer":
        self._lr = LinearRegression().fit(Xc_train, Xm_train)
        return self

    def transform(self, Xc: np.ndarray, Xm: np.ndarray) -> np.ndarray:
        return Xm - self._lr.predict(Xc)
