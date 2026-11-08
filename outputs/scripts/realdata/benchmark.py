#!/usr/bin/env python
"""Step 6: honest benchmark table on the REAL MMRF data.

Classical CoxPH (lifelines) feature-ablation + Weibull AFT + a cytogenetic
subtype-only ("mmSYGNAL-style") model, all on the SAME patient-disjoint split
used by the neural pipeline (mm_tte_survival.data._hash_split, seed 42), with
train-only imputation/scaling and test-set bootstrap CIs. The neural Cox/AFT/
FHT/OPSD rows are read from the repo run's leaderboard.csv and merged in.

No SOTA claims: same endpoint (OS), same split, same C-index metric, CIs shown.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from mm_tte_survival.data import _hash_split
from mm_tte_survival.metrics import harrell_c_index
from lifelines import CoxPHFitter, WeibullAFTFitter

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"
OUTDIR = ROOT / "outputs" / "real_run"
SEED = 42
TIME, EVENT = "time_months", "event"

CLINICAL = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]
CYTO = ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]
N_OMICS_PCS = 32  # cap PCs to control overfitting with limited events


def load_merged():
    clin = pd.read_csv(REAL / "clinical_survival.csv")
    clin["patient_id"] = clin["patient_id"].astype(str)
    df = clin.copy()
    cyto = pd.read_csv(REAL / "cytogenetics.csv")
    cyto["patient_id"] = cyto["patient_id"].astype(str)
    df = df.merge(cyto, on="patient_id", how="left")
    omics_cols = []
    omp = REAL / "omics.csv"
    if omp.exists():
        om = pd.read_csv(omp)
        om["patient_id"] = om["patient_id"].astype(str)
        omics_cols = [c for c in om.columns if c.startswith("PC")][:N_OMICS_PCS]
        df = df.merge(om[["patient_id"] + omics_cols], on="patient_id", how="left")
    df["split"] = df["patient_id"].map(lambda x: _hash_split(x, SEED))
    df["_has_omics"] = df[omics_cols[0]].notna() if omics_cols else False
    return df, omics_cols


def prep_xy(df, cols):
    tr = df["split"].eq("train")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    med = X.loc[tr].median()
    X = X.fillna(med).fillna(0.0)
    mu, sd = X.loc[tr].mean(), X.loc[tr].std().replace(0, 1.0)
    X = (X - mu) / sd
    return X


def fit_cox(df, feat_cols, label):
    X = prep_xy(df, feat_cols)
    d = X.copy()
    d[TIME] = df[TIME].values
    d[EVENT] = df[EVENT].values
    d["split"] = df["split"].values
    tr = d[d.split.eq("train")].drop(columns="split")
    te = d[d.split.eq("test")].drop(columns="split")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tr, duration_col=TIME, event_col=EVENT)
    risk = cph.predict_partial_hazard(te[feat_cols]).values
    return eval_risk(te[TIME].values, te[EVENT].values, risk, label, len(tr))


def fit_aft(df, feat_cols, label):
    X = prep_xy(df, feat_cols)
    d = X.copy()
    d[TIME] = df[TIME].values.clip(min=0.1)
    d[EVENT] = df[EVENT].values
    d["split"] = df["split"].values
    tr = d[d.split.eq("train")].drop(columns="split")
    te = d[d.split.eq("test")].drop(columns="split")
    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(tr, duration_col=TIME, event_col=EVENT)
    # higher predicted median survival = lower risk -> use negative of expectation
    med = aft.predict_median(te[feat_cols]).values
    risk = -med
    return eval_risk(te[TIME].values, te[EVENT].values, risk, label, len(tr))


def eval_risk(t, e, risk, label, n_train, n_boot=500):
    c = harrell_c_index(t, e, risk)
    rng = np.random.default_rng(SEED)
    boots = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if e[idx].sum() < 2:
            continue
        boots.append(harrell_c_index(t[idx], e[idx], risk[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"model": label, "test_cindex": round(float(c), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "n_train": n_train, "n_test": n, "events_test": int(e.sum())}


def main():
    df, omics_cols = load_merged()
    print(f"merged N={len(df)}  has_omics={int(df['_has_omics'].sum())}  "
          f"omics_PCs={len(omics_cols)}", flush=True)
    rows = []
    # ---- Cox feature ablation ----
    rows.append(fit_cox(df, CLINICAL, "Cox: clinical"))
    rows.append(fit_cox(df, CLINICAL + CYTO, "Cox: clinical+cyto"))
    if omics_cols:
        rows.append(fit_cox(df, omics_cols, "Cox: omics(PCs)"))
        rows.append(fit_cox(df, CLINICAL + CYTO + omics_cols, "Cox: clinical+cyto+omics"))
    # ---- classical AFT on full features ----
    full = CLINICAL + CYTO + (omics_cols if omics_cols else [])
    rows.append(fit_aft(df, full, "Weibull-AFT: full (classical)"))
    # ---- mmSYGNAL-style: cytogenetic subtype-only model ----
    rows.append(fit_cox(df, CYTO, "Subtype-only Cox (mmSYGNAL-style)"))

    bench = pd.DataFrame(rows)

    # ---- merge in neural models from repo leaderboard ----
    lb_path = OUTDIR / "leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        neural = lb.rename(columns={"test_cindex_ci_low": "ci_low",
                                    "test_cindex_ci_high": "ci_high"})
        neural["model"] = neural["model"].map(lambda m: f"Neural {m} (repo, full features)")
        neural = neural[["model", "test_cindex", "ci_low", "ci_high", "n_test", "events_test"]]
        bench = pd.concat([bench, neural], ignore_index=True)

    bench = bench.sort_values("test_cindex", ascending=False).reset_index(drop=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bench.to_csv(OUTDIR / "benchmark_table.csv", index=False)

    # markdown
    lines = ["# Benchmark — MMRF-CoMMpass OS (real GDC open data)", "",
             f"Endpoint: overall survival (months). Patient-disjoint hash split (seed {SEED}). "
             f"Same Harrell C-index + 500x test bootstrap CI for all rows.", "",
             "| Model | Test C-index | 95% CI | N test | Events |",
             "|---|---|---|---|---|"]
    for _, r in bench.iterrows():
        ci = f"{r['ci_low']:.3f}–{r['ci_high']:.3f}" if pd.notna(r['ci_low']) else "—"
        nt = int(r['n_test']) if pd.notna(r.get('n_test')) else "—"
        ev = int(r['events_test']) if pd.notna(r.get('events_test')) else "—"
        lines.append(f"| {r['model']} | {r['test_cindex']:.3f} | {ci} | {nt} | {ev} |")
    lines += ["", "Notes:",
              "- OS only; PFS not in GDC open clinical so it is not benchmarked here.",
              "- Cytogenetics are CNV-derived (amp1q/del1p/del13q/del17p/hyperdiploid); "
              "translocations are expression surrogates, not FISH.",
              "- 'mmSYGNAL-style' = cytogenetic subtype-only model, NOT the published "
              "mmSYGNAL regulon programs (not available open-access).",
              "- MyeVAE-style not reproduced (requires published architecture/weights); "
              "omics(PCs) Cox is the closest open analog.",
              "- Wide CIs reflect the small held-out test set; no SOTA claim is made."]
    (OUTDIR / "benchmark_table.md").write_text("\n".join(lines) + "\n")
    print(bench.to_string(index=False))
    print(f"\nwrote {OUTDIR/'benchmark_table.csv'} and .md")


if __name__ == "__main__":
    main()
