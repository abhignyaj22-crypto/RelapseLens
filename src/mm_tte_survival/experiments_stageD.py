"""Stage D — negative-control gate (DECISIVE).

Does HSS learn cytogenetic-subtype BIOLOGY, or is its small-subtype benefit just
shared-trunk regularization? We run the SAME HSS comparison (per-subtype IPCW-IBS,
HSS vs independent-per-subtype) under five conditions on identical patient-disjoint
folds, fully leak-free (in-fold PCA omics + CNV-only subtypes):

  real        - true subtype labels
  permuted    - membership rows shuffled across patients (prevalence + co-occurrence
                preserved, patient<->label link broken)
  random      - fresh Bernoulli memberships at the observed per-subtype prevalence
  lambda0     - real labels, distillation off (lambda=0)
  lambda_huge - real labels, lambda huge (collapse toward agnostic)

Pre-registered decision rule:
  If HSS's small-subtype improvement under REAL labels is ~the same as under
  PERMUTED/RANDOM labels, the benefit is regularization, NOT subtype biology -> STOP
  the subtype-aware novelty claim. Only a real >> permuted/random gap supports it.

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ensure_outdir
from .data.cohort import build_matched_cohort, build_membership_matrix
from .data.gene_expression import load_gene_matrix
from .data.omics_pca import OmicsInFoldPCA
from .data.splits import patient_disjoint_stratified_split, assert_patient_disjoint
from .training.trainer_hss import train_hss, train_single, per_subtype_ibs

TIME, EVENT = "time_months", "event"
CONDITIONS = [  # (name, label_mode, lambda_override or None -> use cfg)
    ("real", "real", None),
    ("permuted", "permuted", None),
    ("random", "random", None),
    ("lambda0", "real", 0.0),
    ("lambda_huge", "real", 50.0),
]


def _membership(M_real, mode, rng):
    if mode == "real":
        return M_real
    if mode == "permuted":
        return M_real[rng.permutation(len(M_real))].copy()
    if mode == "random":
        prev = M_real.mean(axis=0)
        return (rng.random(M_real.shape) < prev).astype("float32")
    raise ValueError(mode)


def _fold_features(df, gene_df, clin, ids, train_ids, k, n_topvar, seed):
    infold = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=seed).fit_transform(
        gene_df, train_ids).reindex(ids)
    infold.columns = [f"inPC{i+1}" for i in range(infold.shape[1])]
    feat = pd.concat([df[clin].reset_index(drop=True), infold.reset_index(drop=True)], axis=1)
    feat = feat.apply(pd.to_numeric, errors="coerce")
    return feat


def run_stage_d(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/stageD"))
    cache = Path(cfg["paths"].get("gene_matrix", "data/real/gene_matrix.npz"))
    n_folds = int(cfg.get("validation", {}).get("stageD_folds", 5))
    base_seed = int(cfg.get("seed", 42))
    k = int(cfg.get("features", {}).get("max_omics_features", 16))
    n_topvar = int(cfg.get("features", {}).get("n_topvar", 2000))

    cfg = {**cfg, "features": {**cfg.get("features", {}), "max_omics_features": k}}
    df, g = build_matched_cohort(cfg)
    gene_df = load_gene_matrix(cache)
    df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    clin, sub_cols = g["clinical"], g["cyto"]          # cyto = CNV-only by config
    M_real = build_membership_matrix(df, sub_cols)
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype("float32")
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values

    all_rows = []
    for cond_name, mode, lam in CONDITIONS:
        cfg_c = {**cfg, "training": {**cfg.get("training", {})}}
        if lam is not None:
            cfg_c["training"]["distill_weight"] = lam
        for f in range(n_folds):
            split = patient_disjoint_stratified_split(df, "patient_id", EVENT, base_seed + f)
            assert_patient_disjoint(ids, np.where(split.values, "train", "test"))
            tr = split.values
            # in-fold features (do not depend on labels); standardize train-only
            feat = _fold_features(df, gene_df, clin, ids, ids[tr], k, n_topvar, base_seed)
            X = feat.fillna(feat[tr].median()).fillna(0.0)
            X = ((X - X[tr].mean()) / X[tr].std().replace(0, 1)).fillna(0.0).values.astype("float32")
            rng = np.random.default_rng(base_seed + 1000 * CONDITIONS.index((cond_name, mode, lam)) + f)
            M = _membership(M_real, mode, rng)
            hss = train_hss(X[tr], M[tr], t[tr], e[tr], X[~tr], M[~tr], t[~tr], e[~tr], cfg_c)

            def make_independent(i, _X=X, _split=split, _M=M):
                mem = _M[:, i] > 0.5
                s_tr, s_va = (_split.values) & mem, (~_split.values) & mem
                # use a slice of train as internal val (per_subtype handles tiny val)
                return train_single(_X[s_tr], t[s_tr], e[s_tr], _X[s_va], t[s_va], e[s_va], cfg_c)

            rows = per_subtype_ibs(hss, X, M, t, e, np.where(tr, "train", "test"),
                                   sub_cols, cfg_c, independent_fn=make_independent)
            for r in rows:
                r["condition"] = cond_name
                r["fold"] = f
            all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    detail.to_csv(outdir / "stageD_detail.csv", index=False)
    ok = detail[detail.status == "ok"].copy()
    ok["improvement"] = ok["ibs_independent"] - ok["ibs_hss"]   # +ve = HSS better
    summ = (ok.groupby(["condition", "subtype"])
            .agg(n_folds=("fold", "nunique"), mean_test_n=("test_n", "mean"),
                 mean_ibs_independent=("ibs_independent", "mean"), mean_ibs_hss=("ibs_hss", "mean"),
                 mean_improvement=("improvement", "mean")).reset_index().round(4))
    summ.to_csv(outdir / "stageD_negative_controls.csv", index=False)
    decision = _decide(summ, sub_cols, M_real)
    _write_decision(outdir, summ, decision)
    return {"outdir": str(outdir), "summary": summ, "decision": decision}


def _decide(summ: pd.DataFrame, sub_cols, M_real) -> dict:
    # small subtypes = the 2 least-prevalent CNV subtypes
    prev = {c: float(M_real[:, i].mean()) for i, c in enumerate(sub_cols)}
    small = [c for c, _ in sorted(prev.items(), key=lambda kv: kv[1])[:2]]
    small_imp = (summ[summ.subtype.isin(small)].groupby("condition")["mean_improvement"]
                 .mean().round(4))
    real = float(small_imp.get("real", np.nan))
    perm = float(small_imp.get("permuted", np.nan))
    rand = float(small_imp.get("random", np.nan))
    margin = 0.01   # IBS units; pre-registered minimum real-vs-control gap
    real_helps = real > 0
    beats_controls = (real - max(perm, rand)) > margin
    verdict = ("BIOLOGY: HSS small-subtype benefit is subtype-specific (real >> permuted/random)"
               if real_helps and beats_controls else
               "REGULARIZATION/NULL: real ~ permuted/random -> benefit is not subtype biology. "
               "STOP the subtype-aware novelty claim per pre-registration.")
    return {"small_subtypes": small, "small_subtype_improvement": small_imp.to_dict(),
            "real": real, "permuted": perm, "random": rand, "margin": margin,
            "real_minus_max_control": round(real - max(perm, rand), 4),
            "passes": bool(real_helps and beats_controls), "verdict": verdict}


def _write_decision(outdir: Path, summ, decision):
    lines = [
        "# Stage D — negative-control decision (open_gdc_os, OS technical validation)", "",
        f"Small (least-prevalent CNV) subtypes: {decision['small_subtypes']}", "",
        "Mean small-subtype IBS improvement (independent - HSS; +ve = HSS better):",
    ]
    for cond in ["real", "permuted", "random", "lambda0", "lambda_huge"]:
        v = decision["small_subtype_improvement"].get(cond)
        if v is not None:
            lines.append(f"  - {cond}: {v:+.4f}")
    lines += [
        "",
        f"real - max(permuted, random) = {decision['real_minus_max_control']:+.4f} "
        f"(pre-registered margin {decision['margin']}).",
        "",
        f"## VERDICT: {'PASS' if decision['passes'] else 'STOP'}",
        decision["verdict"],
        "",
        "- lambda0 vs real isolates the distillation's contribution; lambda_huge is the "
        "collapse-to-agnostic control.",
        "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim.",
    ]
    (outdir / "stageD_decision.md").write_text("\n".join(lines) + "\n")
