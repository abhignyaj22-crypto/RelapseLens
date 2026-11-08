#!/usr/bin/env python3
"""Data-centric accuracy test (DATA_QUALITY.md §3 actions #2+#3): does adding the
standard prognostic labs the clinical arm currently OMITS — LDH, hemoglobin, creatinine,
calcium (CRAB / high-risk markers present in clinical_survival.csv but not in
schema.clinical_cols) — improve the clinical Cox for OS?

Honest protocol: repeated patient-disjoint splits; ALL preprocessing (winsorize caps,
log1p, standardization) fit TRAIN-ONLY per fold (leak-free); Uno IPCW C-index on test;
paired ΔC (expanded − base) with a 95% percentile CI. A null (CI includes 0) is a
legitimate, reportable outcome — we do NOT promote the change into the canonical config
unless the CI clears 0. Exploratory (scripts/analysis/), not a manuscript record.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data/real"
OUT = ROOT / "outputs/analysis"

BASE = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]  # canonical clinical_cols
ADD = ["hemoglobin", "calcium", "ldh", "creatinine"]      # omitted prognostic labs
SKEWED = {"ldh", "creatinine"}                             # winsorize + log1p (skew >5)
SEED, R = 42, 50


def fit_transform_cols(tr: pd.DataFrame, cols):
    """Return (caps, mu, sd) fit on TRAIN only. Winsorize skewed labs to train [p1,p99],
    log1p them, then z-score every column on train stats."""
    caps, work = {}, tr[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        if c in SKEWED:
            lo, hi = work[c].quantile(0.01), work[c].quantile(0.99)
            caps[c] = (lo, hi)
    Z = _apply(work, cols, caps, None, None, medians=work.median())
    return caps, Z.mean(), Z.std().replace(0, 1.0), work.median()


def _apply(df, cols, caps, mu, sd, medians):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(medians)
    for c in cols:
        if c in caps:
            lo, hi = caps[c]
            X[c] = np.log1p(X[c].clip(lo, hi) - min(lo, 0.0))
    if mu is not None:
        X = (X - mu) / sd
    return X


def cox_uno_c(tr, te, cols):
    from lifelines import CoxPHFitter
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
    caps, mu, sd, medians = fit_transform_cols(tr, cols)
    Xtr = _apply(tr, cols, caps, mu, sd, medians)
    Xte = _apply(te, cols, caps, mu, sd, medians)
    d = Xtr.copy(); d["t"] = tr["time_months"].clip(lower=0.1).values; d["e"] = tr["event"].astype(int).values
    cph = CoxPHFitter(penalizer=0.1).fit(d, "t", "e")
    risk = cph.predict_log_partial_hazard(Xte).values
    ytr = Surv.from_arrays(tr["event"].astype(bool), tr["time_months"].clip(lower=0.1))
    yte = Surv.from_arrays(te["event"].astype(bool), te["time_months"].clip(lower=0.1))
    return float(concordance_index_ipcw(ytr, yte, risk)[0])


def main() -> int:
    df = pd.read_csv(REAL / "clinical_survival.csv")
    df = df[df["time_months"].notna() & df["event"].notna()].reset_index(drop=True)
    ev = df["event"].astype(int).values
    C_base, C_exp, dC = [], [], []
    for r in range(R):
        rs = np.random.default_rng(SEED + r)
        # event-stratified patient-disjoint split
        idx0, idx1 = np.where(ev == 0)[0], np.where(ev == 1)[0]
        rs.shuffle(idx0); rs.shuffle(idx1)
        c0, c1 = int(0.7 * len(idx0)), int(0.7 * len(idx1))
        tr_i = np.concatenate([idx0[:c0], idx1[:c1]]); te_i = np.concatenate([idx0[c0:], idx1[c1:]])
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        if te["event"].sum() < 10:
            continue
        cb = cox_uno_c(tr, te, BASE)
        ce = cox_uno_c(tr, te, BASE + ADD)
        C_base.append(cb); C_exp.append(ce); dC.append(ce - cb)
    dC = np.array(dC)
    res = {
        "test": "clinical Cox: base 7 covariates vs base + {LDH, hemoglobin, creatinine, calcium}",
        "n_patients": int(len(df)), "n_splits": len(dC),
        "preprocessing": "winsorize[p1,p99]+log1p on ldh/creatinine, z-score all; ALL fit train-only per fold",
        "C_base_mean": float(np.mean(C_base)), "C_expanded_mean": float(np.mean(C_exp)),
        "dC_mean": float(np.mean(dC)),
        "dC_ci95": [float(np.percentile(dC, 2.5)), float(np.percentile(dC, 97.5))],
        "gate_labs_help": bool(np.percentile(dC, 2.5) > 0),
        "note": "ΔC CI>0 => promote labs into schema.clinical_cols; else pre-registered NULL (do not change headline).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clinical_lab_ablation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
