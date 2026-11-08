"""Matched modeling cohort (patients with all required modalities)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_MAX_OMICS_FEATURES = 16   # default only; override via features.max_omics_features (swept)

# Subtype labels. PRIMARY HSS membership uses CNV-derived calls only (leak-free).
# The 3 translocation labels are RNA-expression surrogates thresholded on
# full-cohort quantiles -> a mild label leak; they are sensitivity/diagnostic only.
CNV_SUBTYPES = ["amp1q", "del1p", "del13q", "del17p", "hyperdiploid"]
RNA_SURROGATE_SUBTYPES = ["t_4_14", "t_11_14", "t_14_16"]


def build_membership_matrix(df: pd.DataFrame, subtype_cols: list[str]) -> np.ndarray:
    """Multi-label cytogenetic membership m in {0,1}^S (NaN/<=0 -> 0). A patient
    may belong to several subtypes; none is forced. Rows with no called
    abnormality are all-zero (pure-agnostic fallback in HSS)."""
    if not subtype_cols:
        return np.zeros((len(df), 0), dtype="float32")
    M = df[subtype_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    return (M > 0).astype("float32")


def _load_tables(cfg: dict):
    p = cfg["paths"]
    clin = pd.read_csv(p["clinical"]); clin["patient_id"] = clin["patient_id"].astype(str)
    cyto = pd.read_csv(p["cytogenetics"]) if p.get("cytogenetics") and Path(p["cytogenetics"]).exists() else None
    omics = pd.read_csv(p["omics"]) if p.get("omics") and Path(p["omics"]).exists() else None
    prog_path = Path(p.get("program_activity") or (Path(p["clinical"]).parent / "program_activity.csv"))
    prog = pd.read_csv(prog_path) if prog_path.exists() else None
    for tbl in (cyto, omics, prog):
        if tbl is not None:
            tbl["patient_id"] = tbl["patient_id"].astype(str)
    return clin, cyto, omics, prog


def build_matched_cohort(cfg: dict):
    """Return (df, groups). Always intersects to all-modality patients so every
    downstream model is compared on the same patients (scientific matched cohort)."""
    schema = cfg["schema"]
    time_col = schema["time_col"]
    max_omics = int(cfg.get("features", {}).get("max_omics_features", DEFAULT_MAX_OMICS_FEATURES))
    clin, cyto, omics, prog = _load_tables(cfg)
    clinical_cols = [c for c in schema.get("clinical_cols", []) if c in clin.columns]
    df = clin.copy()

    cyto_cols = []
    if cyto is not None:
        cyto_cols = [c for c in schema.get("cytogenetic_cols", []) if c in cyto.columns]
        df = df.merge(cyto[["patient_id"] + cyto_cols], on="patient_id", how="left")

    omics_cols = []
    if omics is not None:
        pc_cols = [c for c in omics.columns if c.startswith("PC")]
        cand = pc_cols or [c for c in omics.columns
                           if c != "patient_id" and pd.api.types.is_numeric_dtype(omics[c])]
        omics_cols = cand[:max_omics]
        df = df.merge(omics[["patient_id"] + omics_cols], on="patient_id", how="left")

    prog_cols = []
    if prog is not None:
        prog_cols = [c for c in prog.columns if c != "patient_id" and pd.api.types.is_numeric_dtype(prog[c])]
        df = df.merge(prog[["patient_id"] + prog_cols], on="patient_id", how="left")

    df = df[pd.to_numeric(df[time_col], errors="coerce").gt(0)].copy()
    has_cyto = df[cyto_cols].notna().any(axis=1) if cyto_cols else pd.Series(True, index=df.index)
    has_omics = df[omics_cols].notna().all(axis=1) if omics_cols else pd.Series(True, index=df.index)
    df = df[has_cyto & has_omics].reset_index(drop=True)
    groups = {"clinical": clinical_cols, "cyto": cyto_cols, "omics": omics_cols, "programs": prog_cols}
    return df, groups
