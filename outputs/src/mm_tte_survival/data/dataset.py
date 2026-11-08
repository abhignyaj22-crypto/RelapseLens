from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    t_train: np.ndarray
    e_train: np.ndarray
    ids_train: np.ndarray
    x_val: np.ndarray
    t_val: np.ndarray
    e_val: np.ndarray
    ids_val: np.ndarray
    x_test: np.ndarray
    t_test: np.ndarray
    e_test: np.ndarray
    ids_test: np.ndarray
    feature_names: list[str]
    merged: pd.DataFrame
    subtype_cols: list[str]


def _hash_value(patient_id: str, seed: int = 42) -> float:
    h = hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _hash_split(patient_id: str, seed: int = 42) -> str:
    v = _hash_value(patient_id, seed)
    if v < 0.70:
        return "train"
    if v < 0.85:
        return "val"
    return "test"


def _stratified_event_split(df: pd.DataFrame, id_col: str, event_col: str, seed: int = 42) -> pd.Series:
    """Deterministic patient split that preserves event/censoring balance.

    The original pure hash split is reproducible but can create very small event
    counts in the held-out test set, which is unstable for MM survival studies.
    This split hashes patients *within event strata* so each split receives a
    similar event fraction while remaining patient-disjoint and deterministic.
    """
    split = pd.Series(index=df.index, dtype=object)
    for _, idx in df.groupby(event_col, dropna=False).groups.items():
        idx = list(idx)
        order = sorted(idx, key=lambda i: _hash_value(str(df.loc[i, id_col]), seed))
        n = len(order)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        for j, i in enumerate(order):
            if j < n_train:
                split.loc[i] = "train"
            elif j < n_train + n_val:
                split.loc[i] = "val"
            else:
                split.loc[i] = "test"
    return split


def load_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    p = cfg["paths"]
    clinical = pd.read_csv(p["clinical"])
    cyto = pd.read_csv(p["cytogenetics"]) if p.get("cytogenetics") and Path(p["cytogenetics"]).exists() else None
    omics = pd.read_csv(p["omics"]) if p.get("omics") and Path(p["omics"]).exists() else None
    return clinical, cyto, omics


def _looks_precomputed(cols: list[str]) -> bool:
    """Return True for PC/program/activity matrices that should not be PCA'ed again."""
    if not cols:
        return False
    patterns = [
        re.compile(r"^PC\d+$", re.I),
        re.compile(r"^omics_pca_\d+$", re.I),
        re.compile(r"^program_?\d+$", re.I),
        re.compile(r"^gp_?\d+$", re.I),
        re.compile(r"^module_?\d+$", re.I),
    ]
    hits = sum(any(p.match(c) for p in patterns) for c in cols)
    return hits / max(1, len(cols)) >= 0.80


def prepare_dataset(cfg: dict) -> DatasetBundle:
    schema = cfg["schema"]
    id_col = schema.get("id_col", "patient_id")
    time_col = schema.get("time_col", "time")
    event_col = schema.get("event_col", "event")
    split_col = schema.get("split_col", "split")
    seed = int(cfg.get("seed", 42))
    clinical, cyto, omics = load_tables(cfg)

    required = {id_col, time_col, event_col}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical table missing required columns: {sorted(missing)}")
    if clinical[id_col].duplicated().any():
        raise ValueError("Clinical table must be one row per patient for this TTE task. Aggregate repeated samples before modeling.")

    merged = clinical.copy()
    if split_col not in merged.columns:
        strategy = cfg.get("splitting", {}).get("strategy", "hash")
        if strategy in {"stratified_event", "event_stratified"}:
            merged[split_col] = _stratified_event_split(merged, id_col, event_col, seed)
        else:
            merged[split_col] = merged[id_col].astype(str).map(lambda x: _hash_split(x, seed))

    subtype_cols = [c for c in schema.get("cytogenetic_cols", []) if cyto is not None and c in cyto.columns]
    if cfg.get("features", {}).get("use_cytogenetics", True) and cyto is not None:
        if cyto[id_col].duplicated().any():
            cyto = cyto.groupby(id_col, as_index=False).max(numeric_only=True)
        merged = merged.merge(cyto[[id_col] + subtype_cols], on=id_col, how="left")
        for c in subtype_cols:
            merged[c] = merged[c].fillna(0).astype(float)

    omics_cols: list[str] = []
    if cfg.get("features", {}).get("use_omics", True) and omics is not None:
        if omics[id_col].duplicated().any():
            omics = omics.groupby(id_col, as_index=False).mean(numeric_only=True)
        raw_omics_cols = [c for c in omics.columns if c != id_col and pd.api.types.is_numeric_dtype(omics[c])]
        merged = merged.merge(omics[[id_col] + raw_omics_cols], on=id_col, how="left")
        omics_cols = raw_omics_cols
        if cfg.get("cohort", {}).get("require_omics", False):
            merged = merged.loc[merged[omics_cols].notna().any(axis=1)].copy()
        if cfg.get("features", {}).get("add_omics_missing_indicator", True):
            merged["omics_missing"] = (~merged[omics_cols].notna().any(axis=1)).astype(float)

    if cfg.get("cohort", {}).get("require_cytogenetics", False) and subtype_cols:
        merged = merged.loc[merged[subtype_cols].notna().any(axis=1)].copy()

    # Recompute train mask after any cohort filtering.
    train_mask = merged[split_col].eq("train")
    if not train_mask.any():
        raise ValueError("No training rows after filtering. Relax cohort.require_* options or provide split labels.")

    # Clinical design: use configured one-hot/already-encoded columns if present.
    clinical_cols = [c for c in schema.get("clinical_cols", []) if c in merged.columns]
    feature_cols = []
    if cfg.get("features", {}).get("use_clinical", True):
        feature_cols += clinical_cols
    if cfg.get("features", {}).get("use_cytogenetics", True):
        feature_cols += subtype_cols
    if cfg.get("features", {}).get("use_omics", True) and "omics_missing" in merged.columns and cfg.get("features", {}).get("add_omics_missing_indicator", True):
        feature_cols.append("omics_missing")

    # Omics feature handling.
    # - raw gene/probe matrices: train-only scaling + PCA avoids p>>n instability.
    # - program activity / precomputed PC matrices: pass through; double-PCA can erase
    #   the biological axes the upstream method intentionally constructed.
    if omics_cols:
        omics_mat = merged[omics_cols].astype(float)
        fill = omics_mat.loc[train_mask].median(axis=0)
        omics_filled = omics_mat.fillna(fill).fillna(0.0)
        transform = cfg.get("features", {}).get("omics_transform", "auto")
        if transform == "auto":
            transform = "passthrough" if _looks_precomputed(omics_cols) else "pca"
        if transform == "passthrough":
            merged.loc[:, omics_cols] = omics_filled
            feature_cols += omics_cols
        elif transform == "pca":
            n_comp = int(cfg.get("features", {}).get("omics_pca_components", min(32, len(omics_cols))))
            n_comp = max(1, min(n_comp, len(omics_cols), max(1, train_mask.sum() - 1)))
            scaler_o = StandardScaler().fit(omics_filled.loc[train_mask])
            pca = PCA(n_components=n_comp, random_state=seed).fit(scaler_o.transform(omics_filled.loc[train_mask]))
            z = pca.transform(scaler_o.transform(omics_filled))
            z_cols = [f"omics_pca_{i:03d}" for i in range(n_comp)]
            zdf = pd.DataFrame(z, columns=z_cols, index=merged.index)
            merged = pd.concat([merged, zdf], axis=1)
            feature_cols += z_cols
        else:
            raise ValueError("features.omics_transform must be one of: auto, pca, passthrough")

    if not feature_cols:
        raise ValueError("No feature columns available. Provide clinical, cytogenetic, or omics features.")

    # Numeric coercion and train-only imputation/scaling.
    X_df = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    fill = X_df.loc[train_mask].median(axis=0)
    X_df = X_df.fillna(fill).fillna(0.0)
    scaler = StandardScaler().fit(X_df.loc[train_mask])
    X = scaler.transform(X_df)
    y_t = merged[time_col].astype(float).to_numpy()
    y_e = merged[event_col].astype(int).to_numpy()
    ids = merged[id_col].astype(str).to_numpy()

    def subset(split: str):
        m = merged[split_col].eq(split).to_numpy()
        return X[m], y_t[m], y_e[m], ids[m]

    x_train, t_train, e_train, ids_train = subset("train")
    x_val, t_val, e_val, ids_val = subset("val")
    x_test, t_test, e_test, ids_test = subset("test")
    if len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Split produced empty val/test set. Provide split column or more patients.")
    if e_test.sum() < int(cfg.get("experiments", {}).get("min_test_events", 10)):
        raise ValueError(
            f"Test split has only {int(e_test.sum())} events. Use splitting.strategy=stratified_event, "
            "cross-validation, or a larger endpoint cohort before making model-comparison claims."
        )
    return DatasetBundle(
        x_train=x_train.astype("float32"), t_train=t_train.astype("float32"), e_train=e_train.astype("float32"), ids_train=ids_train,
        x_val=x_val.astype("float32"), t_val=t_val.astype("float32"), e_val=e_val.astype("float32"), ids_val=ids_val,
        x_test=x_test.astype("float32"), t_test=t_test.astype("float32"), e_test=e_test.astype("float32"), ids_test=ids_test,
        feature_names=feature_cols, merged=merged, subtype_cols=subtype_cols,
    )
