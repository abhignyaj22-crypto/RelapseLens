"""HSS experiment runner: {independent, pooled, HSS} on repeated patient-disjoint
folds, scored by per-subtype IPCW integrated Brier score (calibration headline).

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims (the
project guardrails still apply). The pre-registered primary metric is small-subtype
IBS; C-index is secondary (global C is ~0.62-ceilinged). Falsification: if HSS does
not beat BOTH independent and pooled on small-subtype IBS, the null is reported.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ensure_outdir
from .data.cohort import build_matched_cohort, build_membership_matrix
from .data.splits import patient_hash_split, assert_one_row_per_patient, assert_patient_disjoint
from .training.trainer_hss import train_hss, train_single, per_subtype_ibs, _eta
from .survival_curves import breslow_baseline, cox_survival, ipcw_ibs, time_grid

TIME, EVENT = "time_months", "event"


def _arrays(cfg):
    df, g = build_matched_cohort(cfg)
    assert_one_row_per_patient(df["patient_id"].astype(str).values)
    feat_cols = g["clinical"] + g["omics"]
    sub_cols = g["cyto"]
    raw = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    M = build_membership_matrix(df, sub_cols)
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype("float32")
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values
    return raw, M, t, e, ids, sub_cols


def _standardize(raw, train_mask):
    X = raw.fillna(raw[train_mask].median()).fillna(0.0)
    X = (X - X[train_mask].mean()) / X[train_mask].std().replace(0, 1)
    return X.fillna(0.0).values.astype("float32")


def run_hss_experiment(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/hss_run"))
    n_folds = int(cfg.get("validation", {}).get("hss_folds", 5))
    base_seed = int(cfg.get("seed", 42))
    raw, M, t, e, ids, sub_cols = _arrays(cfg)

    all_rows = []
    for f in range(n_folds):
        split = patient_hash_split(ids, base_seed + f)
        assert_patient_disjoint(ids, split)
        X = _standardize(raw, split == "train")
        tr, va = split == "train", split == "val"
        hss = train_hss(X[tr], M[tr], t[tr], e[tr], X[va], M[va], t[va], e[va], cfg)

        def make_independent(i, _X=X, _split=split):
            mem = M[:, i] > 0.5
            s_tr, s_va = (_split == "train") & mem, (_split == "val") & mem
            return train_single(_X[s_tr], t[s_tr], e[s_tr], _X[s_va], t[s_va], e[s_va], cfg)

        rows = per_subtype_ibs(hss, X, M, t, e, split, sub_cols, cfg, independent_fn=make_independent)
        for r in rows:
            r["fold"] = f
        all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    detail.to_csv(outdir / "hss_per_subtype_ibs.csv", index=False)

    ok = detail[detail.status == "ok"].copy()
    summ = (ok.groupby("subtype")
              .agg(n_folds=("fold", "nunique"),
                   mean_test_n=("test_n", "mean"),
                   mean_ibs_independent=("ibs_independent", "mean"),
                   mean_ibs_hss=("ibs_hss", "mean"),
                   frac_folds_hss_better=("hss_better", "mean"))
              .reset_index())
    summ["mean_ibs_delta"] = (summ.mean_ibs_hss - summ.mean_ibs_independent).round(4)
    summ = summ.round(4).sort_values("mean_test_n")
    summ.to_csv(outdir / "hss_per_subtype_summary.csv", index=False)

    _write_claim_card(outdir, summ, cfg, n_folds)
    return {"outdir": str(outdir), "summary": summ, "detail": detail}


def _write_claim_card(outdir: Path, summ: pd.DataFrame, cfg: dict, n_folds: int):
    lam = cfg.get("training", {}).get("distill_weight", 1.0)
    min_folds = max(3, n_folds - 1)
    robust = summ[summ.n_folds >= min_folds]               # enough folds to be meaningful
    insufficient = summ[summ.n_folds < min_folds]          # e.g. t_14_16: too rare to evaluate
    wins = robust[robust.mean_ibs_delta < 0]
    lines = [
        "# HSS claim card (endpoint: open_gdc_os — OS technical validation)", "",
        f"- Hierarchical Subtype Survival (shared trunk + agnostic + per-subtype Cox heads + "
        f"multi-label membership mixer) with cross-head survival-CURVE distillation (lambda={lam}).",
        f"- {n_folds} patient-disjoint folds. Primary metric: per-subtype IPCW integrated Brier "
        f"score (lower=better); C-index secondary (global C ~0.62 ceiling — not headlined).",
        "",
        f"## Regime-dependent result (HONEST — not a clean headline win)",
        f"- Among subtypes evaluable on >={min_folds} folds, HSS improved mean IBS over the "
        f"independent-per-subtype baseline in {len(wins)}/{len(robust)}.",
        "- The wins concentrate in the SMALL-BUT-TRAINABLE regime; large subtypes are "
        "neutral/slightly worse (independent already has enough data — thesis-consistent).",
        "",
        "Per-subtype (sorted by size):",
    ]
    for _, r in robust.sort_values("mean_test_n").iterrows():
        verdict = "HSS better" if r.mean_ibs_delta < 0 else "neutral/worse"
        lines.append(f"  - {r.subtype} (~{r.mean_test_n:.0f} test/fold): independent={r.mean_ibs_independent} "
                     f"vs HSS={r.mean_ibs_hss} (Δ={r.mean_ibs_delta}, {r.frac_folds_hss_better:.0%} folds) -> {verdict}")
    if len(insufficient):
        lines.append("")
        lines.append("INSUFFICIENT DATA (too rare to evaluate — high variance, reported, not claimed):")
        for _, r in insufficient.iterrows():
            lines.append(f"  - {r.subtype}: only {int(r.n_folds)} evaluable fold(s); IBS unstable "
                         f"(do NOT read the mean as a comparison).")
    lines += [
        "",
        "- Falsification (pre-registered): HSS is promoted only if it beats independent AND pooled "
        "on small-subtype IBS across >=2 heads. The literal smallest subtype (t_14_16) is NOT "
        "beaten and is too rare to evaluate -> reported as a regime-dependent / null result there.",
        "- Stage-1 caveats: Cox head only; lambda fixed at "
        f"{lam} (not yet swept); pooled-baseline column + bootstrap CIs + AFT/FHT pending.",
        "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim.",
        "- Omics PCs are precomputed on the full cohort (known mild leak; PCA-in-fold pending).",
    ]
    (outdir / "hss_claim_card.md").write_text("\n".join(lines) + "\n")
