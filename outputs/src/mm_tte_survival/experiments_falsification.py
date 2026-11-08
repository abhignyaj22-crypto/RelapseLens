"""Phase 3 — the decisive falsification matrix (pre-specified in docs/benchmark_protocol.md).

Runs every model arm AND the falsification controls on IDENTICAL patient-disjoint folds
with leak-free in-fold PCA and IPCW metrics, then applies the strict binary control rules:

  * clinical Cox                         -- reference
  * clinical + in-fold omics PCA Cox     -- molecular increment (vs clinical)
  * residual-risk model                  -- omics beyond clinical (vs clinical)
  * clinical + cyto + omics Cox          -- subtype-conditioned linear (vs clinical)
  * pooled neural (MLP-Cox)              -- nonlinear gain (vs clinical+omics)
  * SCRAMBLED-subtype control            -- subtype biology?  (vs real subtype)
  * PERMUTED-omics control  (NEW)        -- molecular signal? (vs real omics)
  * RANDOM survival-label control (NEW)  -- any signal at all? (Uno-C vs 0.5)

Design mirrors evaluate.py: repeated folds give per-arm mean Uno-C / IPCW-IBS (stability),
and a single primary patient-disjoint split gives the paired ΔC (same test patients) with a
paired permutation p-value and BH/Holm multiplicity control.

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims. The full HSS
subtype-conditioned NEURAL arm (biology-vs-regularization) is answered by `stage-d`/`hss`
and is cross-referenced, not duplicated here.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import ensure_outdir
from .data.cohort import build_matched_cohort
from .data.gene_expression import load_gene_matrix
from .data.omics_pca import OmicsInFoldPCA
from .data.splits import patient_disjoint_stratified_split, assert_patient_disjoint
from .comparators import PenalizedCox, MLPCox
from .models.residual_risk import ResidualRiskModel
from .metrics_ipcw import uno_c_index
from .survival_curves import time_grid, breslow_baseline, cox_survival, ipcw_ibs
from .evaluation.stats import paired_delta_cindex, multiplicity_adjust

TIME, EVENT = "time_months", "event"
SCHEMA_VERSION = "1.0"

# claim-safe plot label + provenance for each arm (Fig 3 contract). NEVER "real subtype":
# CNV subtype assignments are sequencing-inferred, not FISH ground truth.
ARM_META = {
    "clinical":            ("Penalized Cox (reference)",           "none",        "reference"),
    "clinical+omics":      ("Clinical + omics Cox",                "none",        "model"),
    "residual_risk":       ("Residual-risk model",                 "none",        "model"),
    "clinical+cyto+omics": ("CNV-derived subtype-conditioned",     "cnv_derived", "observed_inferred"),
    "pooled_neural":       ("Pooled neural survival",              "none",        "model_complexity"),
    "scrambled_subtype":   ("Scrambled-subtype control",           "cnv_derived", "scrambled_control"),
    "permuted_omics":      ("Permuted-omics control",              "none",        "permuted_control"),
    "random_label":        ("Randomized survival-label control",   "none",        "label_randomization"),
}


def _dump(path, obj):
    """JSON dump that tolerates numpy scalars (np.float64/np.bool_/np.int64)."""
    path.write_text(json.dumps(obj, indent=2,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# --------------------------------------------------------------------------- #
# Per-fold leak-free feature construction (real, permuted-omics, scrambled-subtype)
# --------------------------------------------------------------------------- #
def _fold_frame(df, gene_df, clin, cyto, ids, train_ids, k, n_topvar, seed, rng):
    """Return a dict of patient-aligned feature frames for this fold:
    'real' (clin+cyto+inPC), 'perm_omics' (inPC rows shuffled across patients),
    'scram_cyto' (cyto rows shuffled across patients). Omics PCA is fit TRAIN-ONLY."""
    infold = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=seed).fit_transform(gene_df, train_ids).reindex(ids)
    om_cols = [f"inPC{i+1}" for i in range(infold.shape[1])]
    infold.columns = om_cols
    infold = infold.reset_index(drop=True)
    base = df[clin + cyto].reset_index(drop=True)
    real = pd.concat([base, infold], axis=1)
    # permuted-omics: break the omics<->patient(outcome) link, keep the omics distribution
    perm_omics = pd.concat([base, infold.iloc[rng.permutation(len(infold))].reset_index(drop=True)], axis=1)
    # scrambled-subtype: break the cyto<->patient link
    scram = df[clin].reset_index(drop=True)
    scram_cyto_block = df[cyto].reset_index(drop=True)
    scram_cyto_block = scram_cyto_block.iloc[rng.permutation(len(scram_cyto_block))].reset_index(drop=True)
    scram_cyto = pd.concat([scram, scram_cyto_block, infold], axis=1)
    return {"real": real, "perm_omics": perm_omics, "scram_cyto": scram_cyto,
            "om_cols": om_cols}


def _ibs(t_tr, e_tr, eta_tr, t_te, e_te, eta_te) -> float:
    """IPCW integrated Brier via a Breslow baseline from the arm's train linear predictor."""
    grid = time_grid(t_tr, e_tr, t_te)
    if grid is None:
        return float("nan")
    H = breslow_baseline(t_tr, e_tr, eta_tr, grid)
    S = cox_survival(eta_te, H)
    return ipcw_ibs(t_tr, e_tr, t_te, e_te, S, grid)


def _fit_predict(kind, feat, feature_cols, clin, molecular_cols, tr, t, e, seed):
    """Fit an arm on the train rows and return (eta_train, eta_test). All standardization
    is train-only (handled inside the comparator / ResidualRiskModel)."""
    full = feat.copy()
    full[TIME] = t
    full[EVENT] = e
    train_df, test_df = full[tr], full[~tr]
    if kind == "residual":
        m = ResidualRiskModel(clin, molecular_cols, time_col=TIME, event_col=EVENT).fit(train_df)
        return m.predict(train_df).total_risk.values, m.predict(test_df).total_risk.values
    Est = MLPCox if kind == "neural" else PenalizedCox
    comp = Est(feature_cols, time_col=TIME, event_col=EVENT, seed=seed).fit(train_df)
    return comp.risk(train_df), comp.risk(test_df)


def run_falsification(cfg: dict, extra_arms=None) -> dict:
    """Run the frozen 8-arm + 3-control matrix. `extra_arms` (Phase-6 submissions) is an
    optional list of {name, baseline, predict} where predict(train_df, test_df) ->
    (risk_train, risk_test); the DataFrames carry clinical+cyto+inPC{1..k}+time+event, and
    the arm is scored on the SAME folds/metrics/multiplicity as the built-ins."""
    extra_arms = list(extra_arms or [])
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/falsification"))
    cache = Path(cfg["paths"].get("gene_matrix", "data/real/gene_matrix.npz"))
    n_folds = int(cfg.get("validation", {}).get("folds", 5))
    base_seed = int(cfg.get("seed", 42))
    k = int(cfg.get("features", {}).get("omics_k",
            cfg.get("features", {}).get("max_omics_features", 16)))
    n_topvar = int(cfg.get("features", {}).get("n_topvar", 2000))

    cfg = {**cfg, "features": {**cfg.get("features", {}), "max_omics_features": k}}
    df, g = build_matched_cohort(cfg)
    gene_df = load_gene_matrix(cache)
    df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    clin, cyto = g["clinical"], g["cyto"]
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype(float)
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values
    # one fixed label shuffle (paired t,e) for the random-label control
    perm_lbl = np.random.default_rng(base_seed + 99991).permutation(len(df))
    t_rand, e_rand = t[perm_lbl], e[perm_lbl]

    # arm spec: name -> (kind, frame_key, columns_fn, baseline, labels)
    def cols(frame_key, use_cyto, use_om):
        return frame_key, use_cyto, use_om

    ARMS = [
        ("clinical",            "cox",     "real",       False, False, "real",  None),
        ("clinical+omics",      "cox",     "real",       False, True,  "real",  "clinical"),
        ("residual_risk",       "residual","real",       True,  True,  "real",  "clinical"),
        ("clinical+cyto+omics", "cox",     "real",       True,  True,  "real",  "clinical"),
        ("pooled_neural",       "neural",  "real",       False, True,  "real",  "clinical+omics"),
        ("scrambled_subtype",   "cox",     "scram_cyto", True,  True,  "real",  "clinical+cyto+omics"),
        ("permuted_omics",      "cox",     "perm_omics", False, True,  "real",  "clinical+omics"),
        ("random_label",        "cox",     "real",       False, True,  "random", None),
    ]

    all_names = [a[0] for a in ARMS] + [x["name"] for x in extra_arms]
    per_fold = {n: {"uno": [], "ibs": []} for n in all_names}
    primary = {}   # arm -> (t_te, e_te, eta_te) on the primary split (fold 0)
    for f in range(n_folds):
        split = patient_disjoint_stratified_split(df, "patient_id", EVENT, base_seed + f)
        tr = split.values
        assert_patient_disjoint(ids, np.where(tr, "train", "test"))
        rng = np.random.default_rng(base_seed + 7 * f)
        frames = _fold_frame(df, gene_df, clin, cyto, ids, ids[tr], k, n_topvar, base_seed, rng)
        om_cols = frames["om_cols"]
        for (name, kind, fkey, use_cyto, use_om, labels, _base) in ARMS:
            feat = frames[fkey]
            fcols = list(clin) + (list(cyto) if use_cyto else []) + (om_cols if use_om else [])
            mol_cols = (list(cyto) if use_cyto else []) + (om_cols if use_om else [])
            tt, ee = (t_rand, e_rand) if labels == "random" else (t, e)
            eta_tr, eta_te = _fit_predict(kind, feat, fcols, clin, mol_cols, tr, tt, ee, base_seed)
            t_te, e_te = tt[~tr], ee[~tr]
            t_trr, e_trr = tt[tr], ee[tr]
            per_fold[name]["uno"].append(uno_c_index(t_trr, e_trr, t_te, e_te, eta_te))
            per_fold[name]["ibs"].append(_ibs(t_trr, e_trr, eta_tr, t_te, e_te, eta_te))
            if f == 0:
                primary[name] = (t_te, e_te, eta_te)
        # Phase-6 submitted arms: real labels/frame, user callable picks its own columns
        if extra_arms:
            full = frames["real"].copy()
            full[TIME], full[EVENT] = t, e
            full["patient_id"] = ids          # expose id so submitted arms can join static per-patient
            train_df, test_df = full[tr], full[~tr]   # feature blocks (programs/drug/CCLE) leak-free
            for x in extra_arms:
                eta_tr, eta_te = x["predict"](train_df, test_df)
                eta_tr = np.asarray(eta_tr, dtype=float); eta_te = np.asarray(eta_te, dtype=float)
                per_fold[x["name"]]["uno"].append(uno_c_index(t[tr], e[tr], t[~tr], e[~tr], eta_te))
                per_fold[x["name"]]["ibs"].append(_ibs(t[tr], e[tr], eta_tr, t[~tr], e[~tr], eta_te))
                if f == 0:
                    primary[x["name"]] = (t[~tr], e[~tr], eta_te)

    # ---- per-arm leaderboard (mean over folds) ----
    lb = []
    for name in all_names:
        u = [x for x in per_fold[name]["uno"] if np.isfinite(x)]
        b = [x for x in per_fold[name]["ibs"] if np.isfinite(x)]
        lb.append({"arm": name, "n_folds": len(u),
                   "mean_uno_cindex": round(np.mean(u), 4) if u else np.nan,
                   "sd_uno_cindex": round(np.std(u), 4) if u else np.nan,
                   "mean_ipcw_ibs": round(np.mean(b), 4) if b else np.nan})
    leaderboard = pd.DataFrame(lb)

    # ---- paired ΔC on the primary split (same test patients), permutation p + BH/Holm ----
    rows = []
    delta_pairs = [(a[0], a[6]) for a in ARMS] + [(x["name"], x.get("baseline")) for x in extra_arms]
    for name, base in delta_pairs:
        if base is None or base not in primary or name not in primary:
            continue
        t_te, e_te, eta_a = primary[name]
        _, _, eta_b = primary[base]
        rows.append({"comparison": f"{name}_vs_{base}",
                     **paired_delta_cindex(t_te, e_te, eta_a, eta_b, seed=base_seed)})
    deltas = pd.DataFrame(rows)
    if len(deltas):
        pv = deltas["p_permutation"].tolist()
        deltas["p_fdr_bh"] = multiplicity_adjust(pv, "bh")
        deltas["p_holm"] = multiplicity_adjust(pv, "holm")
        deltas["fdr_significant"] = [bool(np.isfinite(x) and x <= 0.05) for x in deltas["p_fdr_bh"]]

    leaderboard.to_csv(outdir / "falsification_matrix.csv", index=False)
    deltas.to_csv(outdir / "falsification_paired_deltas.csv", index=False)
    # long-format per-split table (Fig 3 canonical input): one row per (split, arm, metric)
    ps = []
    for name in all_names:
        label, lsrc, ctype = ARM_META.get(name, (name, "none", "submission"))
        uu, bb = per_fold[name]["uno"], per_fold[name]["ibs"]
        for s in range(max(len(uu), len(bb))):
            for metric, seq in (("uno_cindex", uu), ("ipcw_ibs", bb)):
                if s < len(seq):
                    ps.append({"split_id": s, "arm": name, "plot_label": label,
                               "metric": metric, "value": seq[s], "endpoint": "OS",
                               "label_source": lsrc, "control_type": ctype})
    pd.DataFrame(ps).to_csv(outdir / "falsification_per_split.csv", index=False)
    decision = _decide(leaderboard, deltas)
    _dump(outdir / "falsification_decision.json", decision)
    _write_decision(outdir, leaderboard, deltas, decision)
    # self-describing, versioned benchmark artifact (Phase 5 packaging; tracked in git)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "MMTTE-falsification-open-gdc-os",
        "endpoint": "overall_survival",
        "primary_metric": "uno_ipcw_c_index",
        "provenance": {"n_matched": int(len(df)), "n_folds": n_folds, "seed": base_seed,
                       "omics_k": k, "n_topvar": n_topvar,
                       "clinical_cols": list(clin), "cyto_cols": list(cyto)},
        "leaderboard": leaderboard.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records") if len(deltas) else [],
        "decision": decision,
    }
    _dump(outdir / "benchmark_result.json", artifact)
    return {"outdir": str(outdir), "matrix": leaderboard, "deltas": deltas,
            "decision": decision, "artifact": artifact}


def _sig(deltas, comparison):
    r = deltas[deltas.comparison == comparison]
    if not len(r):
        return None
    return {"delta": float(r.delta_cindex.iloc[0]), "ci_low": float(r.delta_ci_low.iloc[0]),
            "p_perm": float(r.p_permutation.iloc[0]),
            "fdr": float(r.p_fdr_bh.iloc[0]) if "p_fdr_bh" in r else np.nan,
            "fdr_significant": bool(r.fdr_significant.iloc[0]) if "fdr_significant" in r else False}


def _decide(lb: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    U = lb.set_index("arm")["mean_uno_cindex"].to_dict()
    d_om = _sig(deltas, "clinical+omics_vs_clinical")
    d_sub = _sig(deltas, "clinical+cyto+omics_vs_clinical")
    d_neural = _sig(deltas, "pooled_neural_vs_clinical+omics")
    d_scram = _sig(deltas, "scrambled_subtype_vs_clinical+cyto+omics")
    d_perm = _sig(deltas, "permuted_omics_vs_clinical+omics")
    real_vs_perm = round(U.get("clinical+omics", np.nan) - U.get("permuted_omics", np.nan), 4)
    rand_c = U.get("random_label", np.nan)

    findings = {
        "molecular_increment": {
            "delta_omics_vs_clinical": d_om,
            "real_minus_permuted_omics_unoC": real_vs_perm,
            "claim": ("MOLECULAR SIGNAL supported" if (d_om and d_om["fdr_significant"] and real_vs_perm > 0.01)
                      else "NULL: omics increment not robust (CI/FDR fails or real ~ permuted-omics)")},
        "subtype_biology": {
            "delta_subtype_vs_clinical": d_sub, "delta_real_vs_scrambled": d_scram,
            "claim": ("SUBTYPE BIOLOGY supported" if (d_scram and d_scram["fdr_significant"] and d_scram["delta"] > 0)
                      else "NULL: real subtype ~ scrambled -> do not claim subtype biology")},
        "nonlinear_gain": {
            "delta_neural_vs_linear": d_neural,
            "claim": ("NONLINEAR GAIN supported" if (d_neural and d_neural["fdr_significant"] and d_neural["delta"] > 0)
                      else "NULL: neural ~ penalized Cox -> do not claim deep-learning advantage")},
        "pipeline_sanity": {
            "random_label_unoC": rand_c,
            "claim": ("OK: random labels ~ 0.5 (no spurious signal / leakage)"
                      if not np.isfinite(rand_c) or abs(rand_c - 0.5) < 0.1
                      else "WARN: random-label C far from 0.5 -> investigate leakage")},
    }
    any_supported = any("supported" in v["claim"] for k, v in findings.items() if k != "pipeline_sanity")
    findings["overall"] = ("H1 (some structure survives falsification controls)" if any_supported
                           else "H0 retained: no biological structure survives the falsification controls")
    return findings


def _write_decision(outdir: Path, lb, deltas, decision):
    L = ["# Falsification matrix — decision (open_gdc_os, OS technical validation)", "",
         "Pre-specified controls per docs/benchmark_protocol.md. Interpretation is binary.", "",
         "## Arm leaderboard (mean over folds; primary metric Uno IPCW C-index)", "",
         lb.to_string(index=False), "",
         "## Paired ΔC on the primary split (permutation p, BH-FDR)", ""]
    L.append(deltas.to_string(index=False) if len(deltas) else "(no paired deltas)")
    L += ["", "## Verdicts (strict)", ""]
    for key in ["molecular_increment", "subtype_biology", "nonlinear_gain", "pipeline_sanity"]:
        L.append(f"- **{key}**: {decision[key]['claim']}")
    L += ["", f"## OVERALL: {decision['overall']}", "",
          "Cross-reference: the HSS subtype-conditioned NEURAL biology-vs-regularization test "
          "is in `stage-d` (experiments_stageD.py). Endpoint = OS; NO relapse/PFS/clinical-use claim."]
    (outdir / "falsification_decision.md").write_text("\n".join(L) + "\n")
