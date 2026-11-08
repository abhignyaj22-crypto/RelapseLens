"""Claim-drift audit — re-derives verdicts from raw numbers and catches inconsistencies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "autoresearch"))

from claim_audit import audit_rank, audit_conformal, audit_drug, audit_attribution  # noqa: E402


def _rank(verdict="REVERT", ci_low=-0.03, fdr=False, cand_ibs=0.135, base_ibs=0.126,
          rnd=0.485, perm=0.695, cand_uno=0.71, reasons=None):
    r = {"verdict": verdict, "delta_ci_low": ci_low, "fdr_significant": fdr, "cand_ibs": cand_ibs,
         "baseline_ibs": base_ibs, "random_label_uno": rnd, "permuted_omics_uno": perm,
         "cand_uno": cand_uno}
    r["reasons"] = reasons if reasons is not None else {
        "ci_low_gt_0": ci_low > 0, "bh_significant": fdr, "ibs_not_worse": cand_ibs <= base_ibs + 0.005,
        "sanity_random_label": abs(rnd - 0.5) < 0.1, "beats_permuted_omics": cand_uno > perm}
    return r


def test_rank_consistent_passes():
    ok, issues = audit_rank(_rank())
    assert ok and not issues


def test_rank_verdict_drift_caught():
    # numbers say REVERT (ci_low<0) but verdict claims KEEP
    ok, issues = audit_rank(_rank(verdict="KEEP"))
    assert not ok and any("verdict" in i for i in issues)


def test_rank_reason_drift_caught():
    r = _rank()
    r["reasons"]["ci_low_gt_0"] = True          # stored reason contradicts raw delta_ci_low<0
    ok, issues = audit_rank(r)
    assert not ok and any("ci_low_gt_0" in i for i in issues)


def test_conformal_consistent_revert():
    rec = {"nominal": 0.9, "tol": 0.05, "verdict": "REVERT",
           "rule_a_conformal_calibrated_all_strata": False, "rule_b_naive_misses_nominal": True,
           "table": [{"n_events": 190, "conformal_cov_events": 0.868, "naive_cov_events": 0.579},
                     {"n_events": 114, "conformal_cov_events": 0.825, "naive_cov_events": 0.561}]}
    ok, issues = audit_conformal(rec)
    assert ok and not issues


def test_conformal_overclaim_caught():
    # table shows a stratum outside tol, but claims KEEP with rule_a True
    rec = {"nominal": 0.9, "tol": 0.05, "verdict": "KEEP",
           "rule_a_conformal_calibrated_all_strata": True, "rule_b_naive_misses_nominal": True,
           "table": [{"n_events": 114, "conformal_cov_events": 0.825, "naive_cov_events": 0.561}]}
    ok, issues = audit_conformal(rec)
    assert not ok and any("rule_a" in i or "verdict" in i for i in issues)


def test_drug_consistent_capability():
    rec = {"auroc_ci": [0.518, 0.661], "permuted_auroc": 0.451,
           "gate_learnable_signal_above_chance": True, "verdict": "CAPABILITY CONFIRMED"}
    ok, issues = audit_drug(rec)
    assert ok and not issues


def test_drug_overclaim_caught():
    # CI lower bound below 0.5 -> not above chance, but claims CAPABILITY
    rec = {"auroc_ci": [0.48, 0.66], "permuted_auroc": 0.45,
           "gate_learnable_signal_above_chance": True, "verdict": "CAPABILITY CONFIRMED"}
    ok, issues = audit_drug(rec)
    assert not ok


def test_attribution_associational_ok():
    rec = {"causal_gate": {"met": False, "claim_level": "ASSOCIATIONAL ONLY — no causal claim"}}
    ok, issues = audit_attribution(rec)
    assert ok and not issues


def test_attribution_causal_overclaim_caught():
    # gate not met but claims a causal level -> drift
    rec = {"causal_gate": {"met": False, "claim_level": "CAUSAL driver of risk"}}
    ok, issues = audit_attribution(rec)
    assert not ok
