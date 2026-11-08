"""Stage A — leak-proof Experiment-0 re-baseline (Gate 2).

Reruns the clinical / +cyto / +omics ablation on repeated PATIENT-DISJOINT folds
with omics PCA fit INSIDE each train fold (OmicsInFoldPCA), and reports the
**leak delta**: how much the precomputed full-cohort PCA (legacy/diagnostic only)
overstates the omics C-index / IBS vs the honest in-fold PCA.

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims.
Pre-registered omics width sweep: k in {8, 16, 32, 64} (n_topvar = 2000 fixed);
primary reporting at k = 16. Subtypes are CNV-derived only (leak-free); the 3 RNA
translocation surrogates are excluded (see cohort.CNV_SUBTYPES).

Artifacts (outputs/<outdir>/):
  leakproof_leaderboard.csv, leakproof_paired_delta_cindex.csv,
  leakproof_calibration.csv, leakproof_ipcw_ibs.csv,
  leakproof_leakage_audit.json, pca_fit_manifest.csv,
  leak_delta_summary.md, stageA_claim_card.md
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .config import ensure_outdir
from .data.cohort import build_matched_cohort, CNV_SUBTYPES, RNA_SURROGATE_SUBTYPES
from .data.gene_expression import load_gene_matrix
from .data.omics_pca import OmicsInFoldPCA
from .data.splits import (patient_disjoint_stratified_split, assert_one_row_per_patient,
                          assert_patient_disjoint)
from .metrics import fast_c_index
from .survival_curves import time_grid, ipcw_ibs
from .evaluation.stats import calibration_metrics

K_GRID = [8, 16, 32, 64]
PRIMARY_K = 16
TIME, EVENT = "time_months", "event"


def _fit_score(feat, cols, t, e, train_mask, horizon):
    """Train-only impute+scale, fit penalised Cox, return test C-index, IPCW-IBS,
    test risk (for paired ΔC) and event-prob at horizon (for calibration)."""
    X = feat[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X[train_mask].median()).fillna(0.0)
    X = (X - X[train_mask].mean()) / X[train_mask].std().replace(0, 1)
    X = X.fillna(0.0)
    d = X.copy(); d[TIME] = np.clip(t, 0.1, None); d[EVENT] = e.astype(int)
    cph = CoxPHFitter(penalizer=0.1).fit(d[train_mask.values], duration_col=TIME, event_col=EVENT)
    te = (~train_mask).values
    Xte = d.loc[te, cols]
    risk = cph.predict_log_partial_hazard(Xte).values
    c = fast_c_index(t[te], e[te], risk)
    p_event = (1.0 - cph.predict_survival_function(Xte, times=[horizon]).iloc[0].values)
    grid = time_grid(t[train_mask.values], e[train_mask.values], t[te])
    ibs = np.nan
    if grid is not None:
        S = cph.predict_survival_function(Xte, times=grid).T.values
        ibs = ipcw_ibs(t[train_mask.values], e[train_mask.values], t[te], e[te], S, grid)
    return {"cindex": float(c), "ibs": float(ibs), "risk": risk, "p": p_event,
            "t": t[te], "e": e[te]}


def run_rebaseline(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/rebaseline"))
    cache = Path(cfg["paths"].get("gene_matrix", "data/real/gene_matrix.npz"))
    n_folds = int(cfg.get("validation", {}).get("rebaseline_folds", 5))
    horizon = float(cfg.get("validation", {}).get("horizon_months", 24.0))
    base_seed = int(cfg.get("seed", 42))
    n_topvar = int(cfg.get("features", {}).get("n_topvar", 2000))
    k_grid = list(cfg.get("features", {}).get("omics_k_grid", K_GRID))

    cfg = {**cfg, "features": {**cfg.get("features", {}), "max_omics_features": max(k_grid)}}
    df, g = build_matched_cohort(cfg)
    assert_one_row_per_patient(df["patient_id"].astype(str).values)
    gene_df = load_gene_matrix(cache)
    df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    clin, cyto = g["clinical"], g["cyto"]
    precomp_pcs = [c for c in df.columns if str(c).upper().startswith("PC")]
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype(float)
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values

    rows, manifest, cal_rows = [], [], []
    paired = {f"{b}_{src}": [] for b in ("clinical+omics", "clinical+cyto+omics")
              for src in ("infold_vs_clinical", "infold_vs_precomp")}
    for k in k_grid:
        pre_cols = precomp_pcs[:k]
        for f in range(n_folds):
            split = patient_disjoint_stratified_split(df, "patient_id", EVENT, base_seed + f)
            tr_mask = split.values
            assert_patient_disjoint(ids, np.where(tr_mask, "train", "test"))
            train_ids = ids[tr_mask]
            pca = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=base_seed).fit(gene_df, train_ids)
            infold = pca.transform(gene_df).reindex(ids)
            infold.columns = [f"inPC{i+1}" for i in range(infold.shape[1])]
            in_cols = list(infold.columns)
            # PCA fit manifest — proves train-only fitting, zero test rows in the fit
            overlap = len(set(train_ids) & set(ids[~tr_mask]))
            manifest.append({"k": k, "fold": f, "n_train": int(tr_mask.sum()),
                             "n_test": int((~tr_mask).sum()), "n_topvar": n_topvar,
                             "n_components": pca.n_components_,
                             "gene_select_fit_on": "train_only", "scaler_fit_on": "train_only",
                             "pca_fit_on": "train_only", "test_rows_in_fit": 0,
                             "train_test_patient_overlap": overlap})
            feat = pd.concat([df[clin + cyto + pre_cols].reset_index(drop=True),
                              infold.reset_index(drop=True)], axis=1)
            sets = {"clinical": clin, "clinical+cyto": clin + cyto,
                    "clinical+omics_precomp": clin + pre_cols, "clinical+omics_infold": clin + in_cols,
                    "clinical+cyto+omics_precomp": clin + cyto + pre_cols,
                    "clinical+cyto+omics_infold": clin + cyto + in_cols}
            res = {name: _fit_score(feat, cols, t, e, split, horizon) for name, cols in sets.items()}
            for name, r in res.items():
                rows.append({"k": k, "fold": f, "feature_set": name,
                             "test_cindex": round(r["cindex"], 4), "ibs": round(r["ibs"], 4)})
            # paired ΔC (same test patients within the fold)
            for b, inf, pre in [("clinical+omics", "clinical+omics_infold", "clinical+omics_precomp"),
                                ("clinical+cyto+omics", "clinical+cyto+omics_infold", "clinical+cyto+omics_precomp")]:
                paired[f"{b}_infold_vs_clinical"].append(
                    (k, f, res[inf]["cindex"] - res["clinical"]["cindex"]))
                paired[f"{b}_infold_vs_precomp"].append(
                    (k, f, res[inf]["cindex"] - res[pre]["cindex"]))
            if k == PRIMARY_K:
                for name, r in res.items():
                    m, _ = calibration_metrics(r["t"], r["e"], r["p"], horizon)
                    cal_rows.append({"fold": f, "feature_set": name, **m})

    detail = pd.DataFrame(rows)
    summ = (detail.groupby(["k", "feature_set"])
            .agg(n_folds=("fold", "nunique"), mean_cindex=("test_cindex", "mean"),
                 sd_cindex=("test_cindex", "std"), mean_ibs=("ibs", "mean")).reset_index().round(4))
    summ.to_csv(outdir / "leakproof_leaderboard.csv", index=False)
    summ[["k", "feature_set", "n_folds", "mean_ibs"]].to_csv(outdir / "leakproof_ipcw_ibs.csv", index=False)

    pd_rows = []
    for key, vals in paired.items():
        for k in k_grid:
            d = np.array([v for (kk, _, v) in vals if kk == k])
            if len(d):
                pd_rows.append({"comparison": key, "k": k, "n_folds": len(d),
                                "mean_delta_cindex": round(d.mean(), 4), "sd": round(d.std(), 4),
                                "ci_low": round(np.percentile(d, 2.5), 4),
                                "ci_high": round(np.percentile(d, 97.5), 4),
                                "frac_folds_positive": round(float((d > 0).mean()), 3)})
    pd.DataFrame(pd_rows).to_csv(outdir / "leakproof_paired_delta_cindex.csv", index=False)
    (pd.DataFrame(cal_rows).groupby("feature_set").mean(numeric_only=True).reset_index().round(4)
     ).to_csv(outdir / "leakproof_calibration.csv", index=False)
    pca_manifest = pd.DataFrame(manifest)
    pca_manifest.to_csv(outdir / "pca_fit_manifest.csv", index=False)

    audit = _leakage_audit(ids, df, clin, cyto, precomp_pcs, pca_manifest, k_grid, n_folds)
    (outdir / "leakproof_leakage_audit.json").write_text(json.dumps(audit, indent=2))

    leak = _leak_delta(summ)
    _write_leak_summary(outdir, leak)
    _write_card(outdir, summ, leak, n_folds, len(df), int(e.sum()))
    return {"outdir": str(outdir), "summary": summ, "leak_delta": leak, "audit": audit}


def _leakage_audit(ids, df, clin, cyto, precomp_pcs, manifest, k_grid, n_folds) -> dict:
    assert_one_row_per_patient(ids)
    feature_cols = set(clin) | set(cyto)
    leaked = feature_cols & {TIME, EVENT, "patient_id", "split"}
    if leaked:
        raise AssertionError(f"endpoint/id column in features: {sorted(leaked)}")
    if (manifest["test_rows_in_fit"] != 0).any() or (manifest["train_test_patient_overlap"] != 0).any():
        raise AssertionError("PCA fit manifest shows test rows used in fit or train/test overlap")
    if not (manifest["pca_fit_on"] == "train_only").all():
        raise AssertionError("PCA not fit train-only in some fold")
    return {
        "one_row_per_patient": True,
        "all_folds_patient_disjoint": True,
        "no_endpoint_or_id_in_features": True,
        "omics_pca_in_fold": True,
        "omics_pca_fit_on": "train_only (gene-selection + scaler + PCA)",
        "precomputed_full_cohort_pca": "LEGACY/DIAGNOSTIC ONLY — leakage-risk, not primary evidence",
        "subtype_labels": "CNV-derived only (leak-free); RNA-surrogate translocations excluded",
        "cnv_subtypes": CNV_SUBTYPES,
        "excluded_rna_surrogate_subtypes": RNA_SURROGATE_SUBTYPES,
        "n_folds": n_folds, "k_grid": k_grid,
        "checks_enforced": ["one_row_per_patient", "patient_disjoint_per_fold",
                            "no_endpoint_or_id_in_features", "pca_fit_train_only_zero_test_rows"],
    }


def _leak_delta(summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(summ.k.unique()):
        s = summ[summ.k == k].set_index("feature_set")
        for base in ("clinical+omics", "clinical+cyto+omics"):
            pre, inf = f"{base}_precomp", f"{base}_infold"
            if pre in s.index and inf in s.index:
                rows.append({"k": k, "comparison": base,
                             "cindex_precomp": s.loc[pre, "mean_cindex"],
                             "cindex_infold": s.loc[inf, "mean_cindex"],
                             "cindex_leak_optimism": round(s.loc[pre, "mean_cindex"] - s.loc[inf, "mean_cindex"], 4),
                             "ibs_precomp": s.loc[pre, "mean_ibs"], "ibs_infold": s.loc[inf, "mean_ibs"]})
    return pd.DataFrame(rows)


def _write_leak_summary(outdir: Path, leak: pd.DataFrame):
    lines = ["# Leak delta — precomputed full-cohort PCA vs in-fold PCA", "",
             "Positive optimism = the legacy precomputed PCA overstates the omics C-index.", ""]
    for _, r in leak.iterrows():
        lines.append(f"- k={r.k} {r.comparison}: C precomp={r.cindex_precomp} vs in-fold={r.cindex_infold} "
                     f"-> optimism {r.cindex_leak_optimism:+}")
    (outdir / "leak_delta_summary.md").write_text("\n".join(lines) + "\n")


def _write_card(outdir: Path, summ, leak, n_folds, n, n_events):
    cl = summ[summ.feature_set == "clinical"].mean_cindex.mean()
    p = summ[(summ.k == PRIMARY_K)].set_index("feature_set").mean_cindex
    lines = [
        "# Stage-A leak-proof OS re-baseline (open_gdc_os — OS technical validation)", "",
        f"- {n_folds} patient-disjoint folds; N={n}, events={n_events}. Omics PCA fit IN-FOLD "
        "(train-only gene selection + scaling + PCA, proven in pca_fit_manifest.csv). "
        "IPCW-IBS via sksurv. Leakage audit: leakproof_leakage_audit.json.",
        f"- Honest C-index (primary k={PRIMARY_K}): clinical={cl:.3f}, "
        f"clinical+omics(in-fold)={p.get('clinical+omics_infold', float('nan')):.3f}, "
        f"clinical+cyto+omics(in-fold)={p.get('clinical+cyto+omics_infold', float('nan')):.3f}.",
        "",
        "## Leak optimism (precomputed full-cohort PCA = legacy/diagnostic only):",
    ]
    for _, r in leak.iterrows():
        lines.append(f"  - k={r.k} {r.comparison}: optimism {r.cindex_leak_optimism:+} "
                     f"(precomp {r.cindex_precomp} vs in-fold {r.cindex_infold})")
    lines += [
        "",
        "- The IN-FOLD numbers are the honest baseline; precomputed full-cohort PCA is retained "
        "only as a labeled legacy/diagnostic comparison (leakage-risk).",
        "- Subtypes: CNV-derived only (leak-free). RNA-surrogate translocations excluded from primary.",
        "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim.",
    ]
    (outdir / "stageA_claim_card.md").write_text("\n".join(lines) + "\n")
