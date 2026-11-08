"""Required baseline panel.

Every ResistanceMap metric reported publicly must appear in a table with ALL
baselines computed on the IDENTICAL split. A 0.88 -> 0.89 delta over
``per_drug_mean`` is not publishable; a delta over ``deepcdr`` on a
drug-holdout split is.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Regression baselines
# ---------------------------------------------------------------------------
def predict_global_mean(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        target: str) -> np.ndarray:
    mu = float(train_df[target].mean())
    return np.full(len(test_df), mu)


def predict_per_drug_mean(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          target: str) -> np.ndarray:
    """Per-drug mean; falls back to global mean for unseen drugs (drug_holdout)."""
    global_mu = float(train_df[target].mean())
    means = train_df.groupby("drug")[target].mean()
    return test_df["drug"].map(means).fillna(global_mu).to_numpy()


def ridge_on_esm2_meanpool(
    train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Ridge on mean-pooled ESM-2 embeddings. Often matches fancy GNNs at n~150."""
    return Ridge(alpha=alpha).fit(train_emb, train_y).predict(test_emb)


# ---------------------------------------------------------------------------
# Classification baselines
# ---------------------------------------------------------------------------
def predict_base_rate(train_y: np.ndarray, n_test: int) -> np.ndarray:
    p = float(np.asarray(train_y).mean())
    return np.full(n_test, p)


def logistic_on_esm2(
    train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(train_emb, train_y)
    return clf.predict_proba(test_emb)[:, 1]


# ---------------------------------------------------------------------------
# External SOTA reimpl -- DeepCDR-style (structure x expression, deep head)
# ---------------------------------------------------------------------------
def deepcdr_reimpl(train_df, test_df, target: str = "ln_ic50", feature_cols=None):
    """DeepCDR-style drug-response baseline on IDENTICAL splits.

    An honest structural stand-in for DeepCDR (Liu et al. 2020): the paper fuses a drug **graph** (GCN) with
    multi-omics via a deep dense head. Here the same two inputs — molecular **structure** (ECFP4 fingerprint,
    PCA-reduced) and cell **expression** (PCA) — feed a nonlinear **MLP** (the deep head). It is NOT the
    published pooled R²≈0.84 (which sits only ~0.10 over the trivial per-drug-mean floor); it is a real,
    leak-matched number computed on the caller's exact folds. Refuses to fabricate: raises on missing feature
    columns or empty input rather than returning a constant. Callers pass in-fold ``feat_*`` columns (drug
    fingerprint-PCA ⊕ expression-PCA); escalate to a faithful GIN only if this substitute shows real signal.
    """
    if train_df is None or test_df is None:
        raise ValueError("deepcdr_reimpl needs real train/test feature frames — it will not fabricate a number.")
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
    if not feature_cols or target not in train_df.columns:
        raise ValueError(
            "deepcdr_reimpl needs feat_* structure/expression columns and the target; "
            f"found feature_cols={list(feature_cols)[:3]}..., target present={target in getattr(train_df, 'columns', [])}."
        )
    Xtr = train_df[feature_cols].to_numpy(float)
    ytr = train_df[target].to_numpy(float)
    Xte = test_df[feature_cols].to_numpy(float)
    scaler = StandardScaler().fit(Xtr)
    net = MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=400,
                       early_stopping=True, n_iter_no_change=15, random_state=0)
    net.fit(scaler.transform(Xtr), ytr)
    return net.predict(scaler.transform(Xte))


BASELINES_REGRESSION: dict[str, Callable] = {
    "global_mean":    predict_global_mean,
    "per_drug_mean":  predict_per_drug_mean,
    "ridge_esm2":     ridge_on_esm2_meanpool,
    "deepcdr":        deepcdr_reimpl,
}

BASELINES_CLASSIFICATION: dict[str, Callable] = {
    "base_rate":      predict_base_rate,
    "logistic_esm2":  logistic_on_esm2,
    "deepcdr":        deepcdr_reimpl,
}


def require_baseline_row(results_table: pd.DataFrame) -> None:
    """Fail loudly if the results table is missing required baseline rows."""
    required = {"per_drug_mean", "ridge_esm2"}
    have = set(results_table["model"]) if "model" in results_table.columns else set()
    missing = required - have
    if missing:
        raise AssertionError(
            f"Missing required baselines: {missing}. "
            f"No headline publishable without them on the same split."
        )
