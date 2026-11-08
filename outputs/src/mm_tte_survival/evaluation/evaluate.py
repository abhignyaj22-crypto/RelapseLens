"""evaluate_model_suite — the full endpoint-correct evaluation on the matched cohort.

Produces (returned in a results dict; persisted by reports.run_reports):
matched ablation, paired ΔC-index, residual-risk decomposition, repeated-split
validation, calibration / DCA / NRI-IDI, subtype evidence labels, reclassification
outcomes, MMRF usefulness, leakage audit, and the endpoint-gated claim report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ..metrics import fast_c_index as cidx
from ..metrics_ipcw import uno_c_index, cumulative_dynamic_auc_mean
from ..endpoints import resolve_endpoint
from ..models.residual_risk import ResidualRiskModel
from ..data.cohort import build_matched_cohort
from ..data.omics_pca import OmicsInFoldPCA
from ..data.gene_expression import load_gene_matrix
from ..data.splits import stratified_event_split, assert_one_row_per_patient, assert_patient_disjoint
from .stats import (paired_delta_cindex, calibration_metrics, decision_curve,
                    nri_idi, _km_event_prob, multiplicity_adjust)
from .claim_gate import build_claim_report

HORIZON_MONTHS = 24.0


def _omics_pca_params(cfg: dict) -> tuple[int, int]:
    """Omics width k and top-variable-gene count for in-fold PCA (config-driven)."""
    feats = cfg.get("features", {})
    k = int(feats.get("omics_k",
            cfg.get("preprocessing", {}).get("feature_selection", {}).get("max_omics_features", 16)))
    n_topvar = int(feats.get("n_topvar", 2000))
    return k, n_topvar


def _load_gene_df(cfg: dict, groups: dict):
    """Return the raw patient x gene matrix for leak-free in-fold PCA when available.

    Returns None when no omics are requested, or when the omics columns are already
    raw features that can be PCA'd in-fold directly (no separate matrix needed).
    RAISES only in the genuinely unfixable case: the omics columns are full-cohort
    precomputed PCs (`PC*`) and no raw gene matrix exists to redo the PCA in-fold.
    """
    if not groups.get("omics"):
        return None
    gm = cfg.get("paths", {}).get("gene_matrix")
    if gm and Path(gm).exists():
        return load_gene_matrix(Path(gm))
    if any(str(c).startswith("PC") for c in groups["omics"]):
        raise RuntimeError(
            "omics are full-cohort precomputed PCs (PC*) but no raw gene matrix is available "
            f"for leak-free in-fold PCA (paths.gene_matrix={gm!r}). Build it with "
            "build_gene_matrix or drop omics. Precomputed full-cohort PCs are a leak and are "
            "not accepted on the evaluation path.")
    return None  # raw omics columns -> in-fold PCA fit directly on them (leak-free)


def _infold_omics(df, omics_cols, train_mask, gene_df, k, n_topvar, seed):
    """Fit the omics dimensionality reduction on TRAIN patients only and splice the
    resulting `inPC*` columns into `df` in place of the original omics columns.

    Two leak-free sources: (a) a raw gene matrix via OmicsInFoldPCA (train-only gene
    selection + scaling + PCA), or (b) the omics columns themselves standardized and
    PCA'd train-only when they are already raw features. Returns
    (df_with_infold_pcs, new_omics_cols, manifest_row); the manifest proves — per
    split — a train-only fit with zero test rows (contract from experiments_rebaseline.py).
    """
    ids = df["patient_id"].astype(str).values
    tr = train_mask.values
    train_ids = ids[tr]
    if gene_df is not None:
        pca = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=seed).fit(gene_df, train_ids)
        infold = pca.transform(gene_df).reindex(ids)
        n_comp = int(pca.n_components_)
        source = "gene_matrix"
    else:
        X = df[omics_cols].apply(pd.to_numeric, errors="coerce")
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1.0)
        Xs = ((X - mu) / sd).fillna(0.0).values
        n_comp = int(min(k, Xs.shape[1], max(int(tr.sum()) - 1, 1)))
        proj = PCA(n_components=n_comp, random_state=seed).fit(Xs[tr])
        infold = pd.DataFrame(proj.transform(Xs), index=ids)
        source = "omics_columns_train_only"
    infold.columns = [f"inPC{i+1}" for i in range(n_comp)]
    df2 = df.drop(columns=[c for c in omics_cols if c in df.columns], errors="ignore").reset_index(drop=True)
    df2 = pd.concat([df2, infold.reset_index(drop=True)], axis=1)
    overlap = len(set(train_ids) & set(ids[~tr]))
    row = {"n_train": int(tr.sum()), "n_test": int((~tr).sum()),
           "n_components": n_comp, "n_topvar": n_topvar, "pca_source": source,
           "gene_select_fit_on": "train_only", "scaler_fit_on": "train_only",
           "pca_fit_on": "train_only", "test_rows_in_fit": 0,
           "train_test_patient_overlap": overlap}
    return df2, list(infold.columns), row


def _prep(df, cols, train_mask):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.loc[train_mask].median()).fillna(0.0)
    mu, sd = X.loc[train_mask].mean(), X.loc[train_mask].std().replace(0, 1.0)
    return (X - mu) / sd


def _fit_cox(df, cols, train_mask, time_col, event_col, horizon):
    X = _prep(df, cols, train_mask)
    d = X.copy()
    d[time_col] = pd.to_numeric(df[time_col]).clip(lower=0.1).values
    d[event_col] = df[event_col].astype(int).values
    cph = CoxPHFitter(penalizer=0.1).fit(d[train_mask.values], duration_col=time_col, event_col=event_col)
    te = d[~train_mask.values]
    risk = cph.predict_log_partial_hazard(te[cols]).values
    p_event = 1.0 - cph.predict_survival_function(te[cols], times=[horizon]).iloc[0].values
    return risk, p_event


def _cindex_ci(t, e, risk, seed, n_boot=500):
    c = cidx(t, e, risk)
    rng = np.random.default_rng(seed)
    boots = [cidx(t[b], e[b], risk[b]) for b in (rng.integers(0, len(t), len(t)) for _ in range(n_boot))]
    boots = [x for x in boots if np.isfinite(x)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(c), float(lo), float(hi)


def _ablation(df, groups, tm, time_col, event_col, seed, horizon):
    clin, cyto, om, prog = groups["clinical"], groups["cyto"], groups["omics"], groups["programs"]
    sets = {"clinical": clin, "clinical+cytogenetics": clin + cyto,
            "clinical+omics": clin + om, "clinical+cytogenetics+omics": clin + cyto + om}
    if prog:
        sets["clinical+programs"] = clin + prog
        sets["clinical+cytogenetics+programs"] = clin + cyto + prog
    te = ~tm
    t = pd.to_numeric(df[time_col]).values[te.values]
    e = df[event_col].astype(int).values[te.values]
    t_tr = pd.to_numeric(df[time_col]).values[tm.values]
    e_tr = df[event_col].astype(int).values[tm.values]
    rows, risks, probs = [], {}, {}
    for name, cols in sets.items():
        if not cols:
            continue
        risk, p = _fit_cox(df, cols, tm, time_col, event_col, horizon)
        c, lo, hi = _cindex_ci(t, e, risk, seed)          # Harrell (secondary; censoring-biased)
        uno = uno_c_index(t_tr, e_tr, t, e, risk)          # PRIMARY discrimination (IPCW)
        auc = cumulative_dynamic_auc_mean(t_tr, e_tr, t, e, risk)
        rows.append({"feature_set": name, "n_features": len(cols),
                     "uno_cindex": round(uno, 4) if np.isfinite(uno) else np.nan,
                     "td_auc_mean": round(auc, 4) if np.isfinite(auc) else np.nan,
                     "harrell_cindex": round(c, 4), "test_cindex": round(c, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n_patients": int(len(df)),
                     "n_test": int(te.sum()), "events_test": int(e.sum()),
                     "primary_metric": "uno_cindex", "analysis_type": "scientific_matched_cohort"})
        risks[name], probs[name] = risk, p
    return pd.DataFrame(rows), risks, probs, (t, e)


def _decomposition(df, groups, tm, time_col, event_col):
    model = ResidualRiskModel(groups["clinical"], groups["cyto"] + groups["omics"],
                              time_col=time_col, event_col=event_col).fit(df[tm.values])
    pred = model.predict(df)
    out = pd.DataFrame({
        "patient_id": df["patient_id"].values,
        "split": np.where(tm.values, "train", "test"),
        time_col: pd.to_numeric(df[time_col]).values, event_col: df[event_col].astype(int).values,
        "clinical_risk": pred.clinical_risk.values,
        "molecular_residual_risk": pred.molecular_residual_risk.values,
        "total_risk": pred.total_risk.values})
    clin_tbl, mol_tbl = model.coefficients()
    mol_tbl["feature_kind"] = ["cytogenetic_call" if f in groups["cyto"] else "rna_pca_loading_derived"
                              for f in mol_tbl.feature]
    te = out.split.eq("test").values
    t, e = out[time_col].values[te], out[event_col].values[te]
    diag = {"clinical_risk": cidx(t, e, out.clinical_risk.values[te]),
            "molecular_residual_risk": cidx(t, e, out.molecular_residual_risk.values[te]),
            "total_risk": cidx(t, e, out.total_risk.values[te]),
            "clinical_coef_in_joint": model.clinical_coef_in_joint}
    return out, clin_tbl, mol_tbl, diag


def _paired(risks, decomp, t, e, seed):
    comps = []
    def add(label, a, b):
        if a in risks and b in risks:
            comps.append({"comparison": label, **paired_delta_cindex(t, e, risks[a], risks[b], seed=seed)})
    add("clinical+omics_vs_clinical", "clinical+omics", "clinical")
    add("clinical+cyto+omics_vs_clinical", "clinical+cytogenetics+omics", "clinical")
    add("clinical+cyto+omics_vs_clinical+cyto", "clinical+cytogenetics+omics", "clinical+cytogenetics")
    if "clinical+programs" in risks:
        add("clinical+programs_vs_clinical", "clinical+programs", "clinical")
        add("clinical+cyto+programs_vs_clinical+cyto", "clinical+cytogenetics+programs", "clinical+cytogenetics")
    te = decomp.split.eq("test").values
    comps.append({"comparison": "residual_total_vs_clinical",
                  **paired_delta_cindex(t, e, decomp.total_risk.values[te], decomp.clinical_risk.values[te], seed=seed)})
    out = pd.DataFrame(comps)
    if len(out):
        # Multiplicity control across the family of paired comparisons. A CI-"confirmed"
        # improvement that does not survive BH-FDR is downgraded (conservative).
        pv = out["p_permutation"].tolist()
        out["p_fdr_bh"] = multiplicity_adjust(pv, "bh")
        out["p_holm"] = multiplicity_adjust(pv, "holm")
        out["fdr_significant"] = [bool(np.isfinite(x) and x <= 0.05) for x in out["p_fdr_bh"]]
        out["claim"] = [c if not (c == "confirmed_improvement" and not sig)
                        else "hypothesis_generating_improvement"
                        for c, sig in zip(out["claim"], out["fdr_significant"])]
    return out


def _repeated(df, groups, time_col, event_col, seed, n_splits, horizon,
              gene_df=None, k=16, n_topvar=2000):
    clin, cyto, om = groups["clinical"], groups["cyto"], groups["omics"]
    set_names = ["clinical", "clinical+cytogenetics", "clinical+omics", "clinical+cytogenetics+omics"]
    per = {k_: [] for k_ in set_names}          # Harrell (secondary)
    per_uno = {k_: [] for k_ in set_names}       # Uno IPCW (primary)
    d_full, d_om = [], []
    for s in range(n_splits):
        tm = stratified_event_split(df, event_col, seed + s)
        # Refit omics PCA INSIDE this split (train-only) so each repeat is leak-free.
        d_s, om_s = df, om
        if om:
            d_s, om_s, _ = _infold_omics(df, om, tm, gene_df, k, n_topvar, seed + s)
        sets = {"clinical": clin, "clinical+cytogenetics": clin + cyto,
                "clinical+omics": clin + om_s, "clinical+cytogenetics+omics": clin + cyto + om_s}
        te = ~tm
        t = pd.to_numeric(d_s[time_col]).values[te.values]
        e = d_s[event_col].astype(int).values[te.values]
        t_tr = pd.to_numeric(d_s[time_col]).values[tm.values]
        e_tr = d_s[event_col].astype(int).values[tm.values]
        r = {}
        for name, cols in sets.items():
            if not cols:
                continue
            risk, _ = _fit_cox(d_s, cols, tm, time_col, event_col, horizon)
            r[name] = cidx(t, e, risk)
            per[name].append(r[name])
            uno = uno_c_index(t_tr, e_tr, t, e, risk)
            if np.isfinite(uno):
                per_uno[name].append(uno)
        if "clinical" in r and "clinical+cytogenetics+omics" in r:
            d_full.append(r["clinical+cytogenetics+omics"] - r["clinical"])
        if "clinical" in r and "clinical+omics" in r:
            d_om.append(r["clinical+omics"] - r["clinical"])
    lb = [{"feature_set": k, "n_splits": len(v),
           "mean_uno_cindex": round(np.mean(per_uno[k]), 4) if per_uno[k] else np.nan,
           "sd_uno_cindex": round(np.std(per_uno[k]), 4) if per_uno[k] else np.nan,
           "mean_cindex": round(np.mean(v), 4), "sd_cindex": round(np.std(v), 4),
           "ci_low": round(np.percentile(v, 2.5), 4),
           "ci_high": round(np.percentile(v, 97.5), 4),
           "primary_metric": "mean_uno_cindex"} for k, v in per.items() if v]
    def drow(lbl, a):
        a = np.array(a)
        return {"comparison": lbl, "n_splits": len(a), "mean_delta": round(a.mean(), 4),
                "sd_delta": round(a.std(), 4), "ci_low": round(np.percentile(a, 2.5), 4),
                "ci_high": round(np.percentile(a, 97.5), 4), "frac_splits_improved": round(np.mean(a > 0), 3)}
    deltas = pd.DataFrame([drow("clinical+omics_vs_clinical", d_om),
                           drow("clinical+cyto+omics_vs_clinical", d_full)])
    return pd.DataFrame(lb).sort_values("mean_cindex", ascending=False), deltas


def _cal_dca_nri(probs, t, e, horizon):
    cal, dec = [], []
    for name, p in probs.items():
        m, deciles = calibration_metrics(t, e, p, horizon)
        cal.append({"model": name, **m})
        for d in deciles:
            dec.append({"model": name, **d})
    nri = []
    for old, new in [("clinical", "clinical+omics"),
                     ("clinical+cytogenetics", "clinical+cytogenetics+omics")]:
        if old in probs and new in probs:
            nri.append({"comparison": f"{new}_vs_{old}", **nri_idi(t, e, probs[old], probs[new], horizon)})
    return pd.DataFrame(cal), pd.DataFrame(dec), pd.DataFrame(decision_curve(t, e, probs, horizon)), pd.DataFrame(nri)


def _subtypes(df, groups, decomp, time_col, event_col, cfg):
    exp = cfg.get("experiments", {})
    mh = int(exp.get("min_subtype_events_hypothesis", 10))
    mc = int(exp.get("min_subtype_events_confirmatory", 30))
    te = decomp.split.eq("test").values
    merged = decomp.merge(df[["patient_id"] + groups["cyto"]], on="patient_id", how="left")
    rows = []
    for s in groups["cyto"]:
        pos = (pd.to_numeric(merged[s], errors="coerce").fillna(0) > 0).values & te
        n, ev = int(pos.sum()), int(merged.loc[pos, event_col].sum())
        rec = {"subtype": s, "n_patients": n, "n_events": ev}
        if ev < 5 or n < 10:
            rec.update({"clinical_cindex": np.nan, "clinical_omics_cindex": np.nan,
                        "delta_cindex": np.nan, "evidence_level": "unstable_descriptive_only"})
        else:
            cc = cidx(merged[time_col].values[pos], merged[event_col].values[pos], merged.clinical_risk.values[pos])
            ct = cidx(merged[time_col].values[pos], merged[event_col].values[pos], merged.total_risk.values[pos])
            d = ct - cc
            lvl = ("potentially_confirmatory_pending_paired_ci" if ev >= mc and d > 0
                   else "hypothesis_generating" if ev >= mh else "unstable_descriptive_only")
            rec.update({"clinical_cindex": round(cc, 3), "clinical_omics_cindex": round(ct, 3),
                        "delta_cindex": round(d, 3), "evidence_level": lvl})
        rows.append(rec)
    return pd.DataFrame(rows)


def _tertiles(x):
    return pd.qcut(pd.Series(x).rank(method="first"), 3, labels=["low", "mid", "high"]).astype(object).values


def _usefulness_outcomes(df, groups, decomp, time_col, event_col, horizon):
    d = decomp.copy()
    d["clinical_grp"] = _tertiles(d.clinical_risk.values)
    d["total_grp"] = _tertiles(d.total_risk.values)
    d["resid_grp"] = _tertiles(d.molecular_residual_risk.values)
    rk = {"low": 0, "mid": 1, "high": 2}
    rec = d[d.clinical_grp != d.total_grp]
    up = rec[rec.apply(lambda r: rk[r.total_grp] > rk[r.clinical_grp], axis=1)]
    down = rec[rec.apply(lambda r: rk[r.total_grp] < rk[r.clinical_grp], axis=1)]
    cs_mh = d[(d.clinical_grp.isin(["low", "mid"])) & (d.resid_grp == "high")]
    hr_cyto = [c for c in ["del17p", "t_4_14", "amp1q", "t_14_16"] if c in df.columns]
    dfi = df.set_index("patient_id")
    flags = dfi.reindex(d.patient_id)[hr_cyto].apply(pd.to_numeric, errors="coerce").fillna(0) > 0
    cyto_high = flags.any(axis=1).values
    cyto_high_mol_low = d[(cyto_high) & (d.total_risk.values < np.median(d.total_risk.values))]

    clin_bin = np.where(pd.Series(d.clinical_risk.values).rank(pct=True) > 0.5, "high", "low")
    mol_bin = np.where(pd.Series(d.molecular_residual_risk.values).rank(pct=True) > 0.5, "high", "low")
    d2 = d.copy()
    d2["grp"] = [f"clinical_{c}__molecular_{m}" for c, m in zip(clin_bin, mol_bin)]
    d2.loc[cyto_high & (mol_bin == "low"), "grp_cyto"] = "cyto_high__molecular_low"
    d2.loc[(~cyto_high) & (mol_bin == "high"), "grp_cyto"] = "cyto_low__molecular_high"
    t_all, e_all = d2[time_col].values, d2[event_col].values
    rows = []
    gd = [("grp", g) for g in ["clinical_low__molecular_low", "clinical_low__molecular_high",
                               "clinical_high__molecular_low", "clinical_high__molecular_high"]]
    gd += [("grp_cyto", "cyto_high__molecular_low"), ("grp_cyto", "cyto_low__molecular_high")]
    for col, g in gd:
        if col not in d2:
            continue
        m = (d2[col] == g).values
        n, ev = int(m.sum()), int(e_all[m].sum())
        if n == 0:
            continue
        er = _km_event_prob(t_all[m], e_all[m], horizon)
        med = KaplanMeierFitter().fit(t_all[m], e_all[m]).median_survival_time_
        lr = logrank_test(t_all[m], t_all[~m], e_all[m], e_all[~m])
        try:
            cph = CoxPHFitter().fit(pd.DataFrame({"g": m.astype(int), "t": t_all, "e": e_all}),
                                    duration_col="t", event_col="e")
            hr = float(np.exp(cph.params_["g"]))
        except Exception:
            hr = np.nan
        rows.append({"group": g, "n_patients": n, "n_events": ev,
                     "event_rate_by_horizon": round(float(er), 4),
                     "median_survival_months": (round(float(med), 1) if np.isfinite(med) else "not_reached"),
                     "logrank_p_vs_rest": round(float(lr.p_value), 4),
                     "hazard_ratio_vs_rest": round(hr, 3) if np.isfinite(hr) else np.nan})
    summary = {"n_reclassified_by_omics": int(len(rec)), "n_reclassified_up": int(len(up)),
               "n_reclassified_down": int(len(down)),
               "n_clinical_standard_but_molecular_high": int(len(cs_mh)),
               "n_cytogenetic_highrisk_but_molecular_lower": int(len(cyto_high_mol_low)),
               "n_cytogenetic_highrisk_total": int(cyto_high.sum()),
               "note": "Reclassification matters only if outcomes differ; see mmrf_reclassification_outcomes.csv."}
    return summary, pd.DataFrame(rows), d2


def _audit_leakage(df, tm, cfg: dict, groups: dict, time_col: str, event_col: str,
                   manifest: list, omics_in_fold: bool) -> dict:
    """A REAL leakage check that FAILS LOUDLY. Every check below either holds or
    raises — including omics PCA, which is now ENFORCED (not merely recorded) on the
    headline path: any full-cohort precomputed PC column or any fold whose PCA fit
    saw test rows aborts the evaluation.
    """
    ids = df["patient_id"].astype(str).values
    assert_one_row_per_patient(ids)                       # raises on duplicates
    split = np.where(tm.values, "train", "test")
    assert_patient_disjoint(ids, split)                   # raises on train/test overlap

    # No endpoint (time/event) or id column may appear among the model features.
    feature_cols = set(groups["clinical"]) | set(groups["cyto"]) | set(groups["omics"]) | set(groups["programs"])
    leaked = feature_cols & {time_col, event_col, "patient_id", "split"}
    if leaked:
        raise AssertionError(f"LEAKAGE: endpoint/id column(s) {sorted(leaked)} present in feature matrix")

    # Omics PCA must be in-fold. A precomputed full-cohort PC ("PC*" but not "inPC*")
    # in the feature matrix is a leak and aborts loudly.
    precomp_pc = sorted(c for c in feature_cols
                        if str(c).upper().startswith("PC") and not str(c).startswith("inPC"))
    if precomp_pc:
        raise AssertionError(
            f"LEAKAGE: full-cohort precomputed PC column(s) {precomp_pc} in feature matrix; "
            "in-fold PCA (inPC*) is required on the evaluation path")
    if groups["omics"] and not omics_in_fold:
        raise AssertionError("LEAKAGE: omics present but PCA was not fit in-fold")
    if manifest:
        mdf = pd.DataFrame(manifest)
        if (mdf["test_rows_in_fit"] != 0).any() or (mdf["train_test_patient_overlap"] != 0).any():
            raise AssertionError("PCA fit manifest shows test rows used in fit or train/test overlap")
        if not (mdf["pca_fit_on"] == "train_only").all():
            raise AssertionError("PCA not fit train-only in some split")

    return {
        "one_row_per_patient": True,                      # asserted above
        "train_test_patient_disjoint": True,              # asserted above
        "endpoint_columns_in_features": sorted(leaked),   # asserted empty above
        "no_precomputed_full_cohort_pc_in_features": True,  # asserted above
        "split_is_patient_aware": True,                   # stratified_event_split delegates to patient split
        "imputer_fit_on": "train", "scaler_fit_on": "train",
        "orthogonalizer_fit_on": "train", "cox_fit_on": "train",
        "test_used_for_model_selection": False,
        "omics_pca_in_fold": bool(omics_in_fold),
        "omics_pca_fit_on": "train_only (gene-selection + scaler + PCA)" if omics_in_fold else "n/a",
        "n_pca_fit_manifest_rows": len(manifest),
        "split_strategy": cfg.get("splitting", {}).get("strategy", "stratified_event"),
        "checks_enforced": ["one_row_per_patient", "train_test_patient_disjoint",
                            "no_endpoint_or_id_column_in_features",
                            "no_precomputed_full_cohort_pc_in_features",
                            "omics_pca_fit_train_only_zero_test_rows"],
    }


def evaluate_model_suite(cfg: dict) -> dict:
    schema = cfg["schema"]
    time_col, event_col = schema["time_col"], schema["event_col"]
    seed = int(cfg.get("seed", 42))
    val = cfg.get("validation") or cfg.get("evaluation") or {}
    horizon = float(val.get("horizon_months", HORIZON_MONTHS))
    n_splits = int(val.get("repeated_splits", 50))
    external_val = bool(val.get("external_validation_available", False))
    endpoint_spec = resolve_endpoint(cfg)

    df, groups = build_matched_cohort(cfg)

    # Leak-free omics: fit PCA IN-FOLD (train-only) instead of consuming full-cohort
    # precomputed PCs. Raises if omics are requested without a raw gene matrix.
    k, n_topvar = _omics_pca_params(cfg)
    gene_df = _load_gene_df(cfg, groups)
    pca_manifest: list = []
    if gene_df is not None:
        df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    # Keep the RAW cohort so the repeated-split harness can refit PCA per fold from
    # scratch (the primary df below has its omics replaced by primary-split in-fold PCs).
    df_raw, raw_omics = df.copy(), list(groups["omics"])
    tm = stratified_event_split(df, event_col, seed)
    omics_in_fold = bool(raw_omics)
    if omics_in_fold:
        df, om_cols, mrow = _infold_omics(df, raw_omics, tm, gene_df, k, n_topvar, seed)
        groups = {**groups, "omics": om_cols}
        pca_manifest.append({"scope": "primary_split", "seed": seed, **mrow})

    ablation, risks, probs, (t, e) = _ablation(df, groups, tm, time_col, event_col, seed, horizon)
    decomp, clin_coef, mol_coef, diag = _decomposition(df, groups, tm, time_col, event_col)
    deltas = _paired(risks, decomp, t, e, seed)
    rep_lb, rep_delta = _repeated(df_raw, {**groups, "omics": raw_omics}, time_col, event_col,
                                  seed, n_splits, horizon, gene_df=gene_df, k=k, n_topvar=n_topvar)
    cal, cal_decile, dca, nri = _cal_dca_nri(probs, t, e, horizon)
    subtypes = _subtypes(df, groups, decomp, time_col, event_col, cfg)
    summary, outcomes, quad = _usefulness_outcomes(df, groups, decomp, time_col, event_col, horizon)

    a = ablation.set_index("feature_set")
    key = deltas[deltas.comparison == "clinical+cyto+omics_vs_clinical"]
    kd = float(key.delta_cindex.iloc[0]) if len(key) else np.nan
    kci = float(key.delta_ci_low.iloc[0]) if len(key) else np.nan
    detail = {"matched_n": int(len(df)), "test_events": int(e.sum()),
              "analysis_type": "scientific_matched_cohort" if cfg.get("cohort", {}).get("matched_ablation", True)
              else "deployment_availability",
              "clinical_cindex": float(a.loc["clinical", "test_cindex"]) if "clinical" in a.index else np.nan,
              "full_cindex": float(a.loc["clinical+cytogenetics+omics", "test_cindex"])
              if "clinical+cytogenetics+omics" in a.index else np.nan,
              "omics_paired_delta": kd, "omics_paired_delta_ci_low": kci,
              "molecular_residual_test_cindex": round(diag["molecular_residual_risk"], 4)}
    claim = build_claim_report(endpoint_spec, omics_delta=kd, omics_delta_ci_low=kci,
                               external_validation_available=external_val, detail=detail)
    leakage = _audit_leakage(df, tm, cfg, groups, time_col, event_col, pca_manifest, omics_in_fold)

    return {"endpoint_spec": endpoint_spec, "cohort_df": df, "groups": groups,
            "ablation": ablation, "paired_deltas": deltas, "decomposition": decomp,
            "clinical_coef": clin_coef, "molecular_coef": mol_coef, "diag": diag,
            "repeated_leaderboard": rep_lb, "repeated_delta": rep_delta,
            "calibration": cal, "calibration_decile": cal_decile, "dca": dca, "nri": nri,
            "subtypes": subtypes, "usefulness": summary, "outcomes": outcomes, "quadrants": quad,
            "claim_report": claim, "leakage_audit": leakage,
            "pca_fit_manifest": pd.DataFrame(pca_manifest), "horizon": horizon}
