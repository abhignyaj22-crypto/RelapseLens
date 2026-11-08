"""Phase 6 — new-arm submission harness for the MMTTE falsification benchmark.

Score ANY survival model on the IDENTICAL patient-disjoint folds, in-fold PCA, IPCW
metrics, and multiplicity correction as the frozen benchmark, then judge it under the same
honesty gate. Submission contract: a callable ``predict(train_df, test_df) -> (risk_train,
risk_test)`` where the DataFrames carry clinical + cyto + inPC{1..k} + time + event columns
(higher risk = earlier event). A submission is written to
``<outdir>/submissions/<name>/`` and NEVER clobbers the frozen canonical artifact.

No submission can be SOTA-licensed on open data: external validation is unavailable, so the
honesty gate refuses superiority prose by construction.
"""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np

from .experiments_falsification import run_falsification, _dump, TIME, EVENT
from .evaluation.claim_gate import benchmark_honesty_lint


def feature_predict(comparator_factory):
    """Adapt a comparator factory (``feature_cols -> estimator`` with ``fit/risk``) to the
    submission contract, using ALL non-label columns (clinical+cyto+omics) as features."""
    def predict(train_df, test_df):
        cols = [c for c in train_df.columns if c not in (TIME, EVENT)]
        est = comparator_factory(cols).fit(train_df)
        return est.risk(train_df), est.risk(test_df)
    return predict


def submission_verdict(row: dict, delta: dict, lint: dict) -> dict:
    """Pure judgement: an arm is credited only if it beats its baseline (ΔCI_low>0) AND is
    FDR-significant; SOTA language is licensed only if the honesty gate says so (never on
    open data). No pipeline / IO here."""
    delta = delta or {}
    ci_low = delta.get("delta_ci_low", float("nan"))
    fdr_sig = bool(delta.get("fdr_significant"))
    ci_pos = bool(np.isfinite(ci_low) and ci_low > 0)
    credited = fdr_sig and ci_pos
    sota = bool(lint.get("licensed_for_sota_language", False))
    if credited and sota:
        msg = "CREDITED and SOTA-licensed."
    elif credited:
        msg = ("PROVISIONAL improvement (beats baseline CI + FDR) but NOT SOTA-licensed: "
               "external validation is unavailable on open data.")
    else:
        msg = "No evidence of improvement over baseline (CI overlaps 0 or fails FDR)."
    return {"credited": credited, "beats_baseline_ci": ci_pos, "fdr_significant": fdr_sig,
            "sota_licensed": sota, "message": msg}


def submit_arm(cfg: dict, name: str, predict, baseline: str = "clinical+omics",
               persist: bool = True) -> dict:
    """Run one submitted arm through the frozen matrix and return its scored verdict."""
    cfg2 = copy.deepcopy(cfg)
    base_out = cfg2.get("paths", {}).get("outdir", "outputs/falsification")
    cfg2.setdefault("paths", {})["outdir"] = str(Path(base_out) / "submissions" / name)
    res = run_falsification(cfg2, extra_arms=[{"name": name, "baseline": baseline, "predict": predict}])
    lb = res["matrix"].to_dict("records")
    deltas = res["deltas"].to_dict("records") if len(res["deltas"]) else []
    row = next((r for r in lb if r["arm"] == name), {})
    delta = next((d for d in deltas if d["comparison"] == f"{name}_vs_{baseline}"), {})
    evidence = {"matched_cohort": True, "same_endpoint": True,
                "delta_ci_low": delta.get("delta_ci_low", float("nan")),
                "fdr_significant": bool(delta.get("fdr_significant")),
                "external_validation": False}
    lint = benchmark_honesty_lint(f"{name} beats state-of-the-art on MM survival.", evidence)
    out = {"name": name, "baseline": baseline, "row": row, "delta": delta,
           "honesty": lint, "verdict": submission_verdict(row, delta, lint),
           "outdir": res["outdir"]}
    if persist:
        _dump(Path(res["outdir"]) / "submission.json", out)
    return out


def demo_coxnet_submission(cfg: dict) -> dict:
    """Runnable example: submit an sksurv Coxnet arm (all features) vs clinical+omics."""
    from .comparators import CoxnetC
    return submit_arm(cfg, "coxnet_all_features", feature_predict(CoxnetC),
                      baseline="clinical+omics")
