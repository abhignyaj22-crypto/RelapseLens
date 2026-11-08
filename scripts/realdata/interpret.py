#!/usr/bin/env python
"""Step 7: interpretation on the REAL MMRF data.

Produces (all on the held-out test split unless noted):
  - permutation feature importance (drop in Harrell C when a feature is shuffled)
  - subtype-specific feature importance (cohort-level Cox |coef| within subtypes)
  - KM curves by predicted-risk tertile + log-rank p   -> figures/km_risk_groups.png
  - calibration: observed vs predicted risk decile      -> figures/calibration.png
  - PC->gene drivers for the most prognostic omics PCs   -> pc_gene_drivers.csv
    (a transparent substitute for pathway enrichment; feed to gseapy/Enrichr later)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from mm_tte_survival.data import _hash_split
from mm_tte_survival.metrics import harrell_c_index

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"
OUT = ROOT / "outputs" / "real_run"
FIG = OUT / "figures"
SEED, TIME, EVENT = 42, "time_months", "event"
CLINICAL = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]
CYTO = ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]


def load():
    clin = pd.read_csv(REAL / "clinical_survival.csv"); clin["patient_id"] = clin["patient_id"].astype(str)
    cyto = pd.read_csv(REAL / "cytogenetics.csv"); cyto["patient_id"] = cyto["patient_id"].astype(str)
    df = clin.merge(cyto, on="patient_id", how="left")
    df["split"] = df["patient_id"].map(lambda x: _hash_split(x, SEED))
    return df


def prep(df, cols):
    tr = df["split"].eq("train")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.loc[tr].median()).fillna(0.0)
    mu, sd = X.loc[tr].mean(), X.loc[tr].std().replace(0, 1.0)
    return (X - mu) / sd


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    feats = CLINICAL + CYTO
    X = prep(df, feats)
    d = X.copy(); d[TIME] = df[TIME].values; d[EVENT] = df[EVENT].values; d["split"] = df["split"].values
    tr = d[d.split.eq("train")].drop(columns="split")
    te = d[d.split.eq("test")].drop(columns="split")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tr, duration_col=TIME, event_col=EVENT)
    risk = cph.predict_partial_hazard(te[feats]).values
    t, e = te[TIME].values, te[EVENT].values
    base_c = harrell_c_index(t, e, risk)
    print(f"test Cox C-index = {base_c:.3f}  (N_test={len(t)}, events={int(e.sum())})")

    # ---- permutation importance ----
    rng = np.random.default_rng(SEED)
    rows = []
    Xte = te[feats].copy()
    for f in feats:
        drops = []
        for _ in range(20):
            Xp = Xte.copy(); Xp[f] = rng.permutation(Xp[f].values)
            rp = cph.predict_partial_hazard(Xp).values
            drops.append(base_c - harrell_c_index(t, e, rp))
        rows.append({"feature": f, "importance_dropC": float(np.mean(drops)),
                     "coef": float(cph.params_[f]), "exp_coef_HR": float(np.exp(cph.params_[f]))})
    imp = pd.DataFrame(rows).sort_values("importance_dropC", ascending=False)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    print("\nTop permutation-importance features:")
    print(imp.head(8).to_string(index=False))

    # ---- subtype-specific importance: |coef| of clinical Cox fit WITHIN each subtype+ group ----
    sub_rows = []
    for s in CYTO:
        mask = pd.to_numeric(df[s], errors="coerce").fillna(0) > 0
        sub = df[mask]
        if sub[EVENT].sum() < 10 or len(sub) < 30:
            sub_rows.append({"subtype": s, "n": int(len(sub)), "events": int(sub[EVENT].sum()), "status": "too_sparse"})
            continue
        Xs = prep(sub.assign(split="train"), CLINICAL)
        ds = Xs.copy(); ds[TIME] = sub[TIME].values; ds[EVENT] = sub[EVENT].values
        try:
            c = CoxPHFitter(penalizer=0.5).fit(ds, duration_col=TIME, event_col=EVENT)
            top = c.params_.abs().idxmax()
            sub_rows.append({"subtype": s, "n": int(len(sub)), "events": int(sub[EVENT].sum()),
                             "top_clinical_driver": top, "coef": float(c.params_[top]), "status": "ok"})
        except Exception as ex:
            sub_rows.append({"subtype": s, "n": int(len(sub)), "events": int(sub[EVENT].sum()), "status": f"err:{ex}"[:40]})
    pd.DataFrame(sub_rows).to_csv(OUT / "subtype_feature_importance.csv", index=False)

    # ---- KM by predicted-risk tertile (test set) ----
    g = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["low", "mid", "high"])
    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    for lab in ["low", "mid", "high"]:
        m = (g == lab).values
        if m.sum() == 0:
            continue
        kmf.fit(t[m], e[m], label=f"{lab} risk (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    lr = multivariate_logrank_test(t, g, e)
    ax.set_title(f"OS by predicted-risk tertile (log-rank p={lr.p_value:.1e})")
    ax.set_xlabel("months"); ax.set_ylabel("survival probability")
    fig.tight_layout(); fig.savefig(FIG / "km_risk_groups.png", dpi=130); plt.close(fig)
    print(f"\nKM log-rank p = {lr.p_value:.2e}")

    # ---- calibration: observed event rate vs predicted-risk decile ----
    order = np.argsort(risk)
    dec = np.array_split(order, 10)
    pred, obs = [], []
    for idx in dec:
        pred.append(np.mean(risk[idx]))
        obs.append(np.mean(e[idx]))  # observed event fraction (OS death) in test follow-up
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(np.argsort(np.argsort(pred)) + 1, obs, "o-")
    ax.set_xlabel("predicted-risk decile (low→high)"); ax.set_ylabel("observed death fraction (test)")
    ax.set_title("Calibration by risk decile"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=130); plt.close(fig)

    # ---- PC -> gene drivers (proxy for pathway enrichment) ----
    drivers_path = REAL / "omics_pc_loadings.csv"
    if drivers_path.exists():
        load_df = pd.read_csv(drivers_path, index_col=0)
        # rank PCs by univariate association with risk on test
        omics = pd.read_csv(REAL / "omics.csv"); omics["patient_id"] = omics["patient_id"].astype(str)
        merged = df.merge(omics, on="patient_id", how="inner")
        merged = merged[merged.split.eq("test")] if (merged.split.eq("test")).any() else merged
        pc_cols = [c for c in omics.columns if c.startswith("PC") and c in merged.columns]
        assoc = []
        for pc in pc_cols:
            if merged[pc].notna().sum() > 20:
                assoc.append((pc, abs(np.corrcoef(merged[pc].fillna(0), merged[EVENT])[0, 1])))
        assoc.sort(key=lambda x: -x[1])
        out_rows = []
        for pc, a in assoc[:5]:
            if pc in load_df.columns:
                top_pos = load_df[pc].sort_values(ascending=False).head(15)
                for gene, w in top_pos.items():
                    out_rows.append({"pc": pc, "event_corr": round(a, 3), "gene": gene, "loading": round(float(w), 4)})
        pd.DataFrame(out_rows).to_csv(OUT / "pc_gene_drivers.csv", index=False)
        print(f"wrote pc_gene_drivers.csv ({len(out_rows)} gene rows)")

    print(f"\nFigures in {FIG}")


if __name__ == "__main__":
    main()
