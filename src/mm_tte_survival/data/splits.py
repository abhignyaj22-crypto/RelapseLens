"""Patient-disjoint splitting — one official splitter for the active pipeline."""
from __future__ import annotations

import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _hash_frac(value: str, seed: int = 42) -> float:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def hash_split(patient_id: str, seed: int = 42) -> str:
    """Deterministic train/val/test by hashing the patient id."""
    v = _hash_frac(patient_id, seed)
    return "train" if v < 0.70 else ("val" if v < 0.85 else "test")


def patient_disjoint_stratified_split(df: pd.DataFrame, patient_col: str, event_col: str,
                                      seed: int, *, mode: str = "train_test",
                                      test_frac: float = 0.25, val_frac: float = 0.0,
                                      event_rate_tol: float = 0.08):
    """THE official splitter: patient-disjoint AND event-rate-preserving.

    Patients are hashed deterministically WITHIN event strata (patient-level
    event-ever), so every split gets a similar event fraction while no patient
    ever crosses splits. Works on one-row-per-patient and longitudinal frames.

    mode="train_test"      -> boolean train mask (Series), drop-in for old callers.
    mode="train_val_test"  -> Series of {train,val,test} labels.
    Asserts patient-disjointness and approximate event-rate preservation.
    """
    pe = df.groupby(patient_col)[event_col].max().astype(int)   # patient -> event-ever
    label_by_patient: dict = {}
    for ev, grp in pe.groupby(pe):
        pats = sorted(grp.index, key=lambda p: _hash_frac(str(p), seed))
        n = len(pats)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n)) if mode == "train_val_test" else 0
        for j, p in enumerate(pats):
            label_by_patient[p] = ("test" if j < n_test
                                   else "val" if j < n_test + n_val else "train")
    split = df[patient_col].map(label_by_patient)
    assert_patient_disjoint(df[patient_col].values, split.values)
    _assert_event_rate_preserved(df, split.values, event_col, event_rate_tol)
    if mode == "train_val_test":
        return split
    return pd.Series((split == "train").values, index=df.index)


def stratified_event_split(df: pd.DataFrame, event_col: str, seed: int,
                           test_frac: float = 0.25, patient_col: str = "patient_id") -> pd.Series:
    """Boolean train mask. Delegates to the patient-disjoint splitter when a
    patient column is present (the active path); the old row-indexed fallback is
    quarantined to patient-less frames and warns."""
    if patient_col in df.columns:
        return patient_disjoint_stratified_split(df, patient_col, event_col, seed,
                                                 mode="train_test", test_frac=test_frac)
    warnings.warn("row-indexed split (no patient column); NOT patient-aware — "
                  "for longitudinal data provide patient_col.", stacklevel=2)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed,
                              stratify=df[event_col].astype(int).values)
    split = np.array(["train"] * len(df), dtype=object)
    split[te] = "test"
    return pd.Series(split == "train")


def _assert_event_rate_preserved(df, split, event_col, tol: float, min_n: int = 30) -> None:
    """Catch a broken (non-stratified) split. Skipped for small partitions where
    rounding dominates — the stratified construction preserves the rate by design."""
    e = df[event_col].astype(int).values
    overall = e.mean()
    for part in ("train", "test"):
        m = np.asarray(split) == part
        if m.sum() >= min_n and abs(e[m].mean() - overall) > tol:
            raise AssertionError(f"event rate in {part} ({e[m].mean():.3f}) deviates from "
                                 f"overall ({overall:.3f}) by > {tol}")


def patient_hash_split(ids, seed: int = 42) -> np.ndarray:
    """Deterministic patient-disjoint 3-way split (train/val/test) by hashing the
    patient id. The same patient always lands in the same split regardless of how
    many rows they have — the patient-aware splitter."""
    return np.array([hash_split(str(p), seed) for p in ids], dtype=object)


def assert_one_row_per_patient(ids) -> None:
    s = pd.Series(list(ids))
    if s.duplicated().any():
        dups = s[s.duplicated()].unique()[:5]
        raise AssertionError(f"Expected one row per patient; found duplicates e.g. {list(dups)}")


def assert_patient_disjoint(ids, split) -> None:
    ids, split = np.asarray(ids).astype(str), np.asarray(split)
    train_ids = set(ids[split == "train"])
    for other in ("val", "test"):
        overlap = train_ids & set(ids[split == other])
        if overlap:
            raise AssertionError(f"LEAKAGE: {len(overlap)} patient(s) in BOTH train and {other}, "
                                 f"e.g. {list(overlap)[:5]}")
