"""Backward-compatibility shim.

The residual-risk monolith has been refactored into clean layers:
  - data.cohort / data.splits        (matched cohort + splitting)
  - models.residual_risk             (ResidualRiskModel)
  - evaluation.evaluate              (evaluate_model_suite)
  - evaluation.stats / claim_gate    (statistics + claim gating)
  - reports.run_reports              (write_all_reports + cards)

`run_residual_report` is retained here so existing callers / the `residual-report`
CLI keep working; it simply evaluates the suite and writes all reports.
"""
from __future__ import annotations

from .evaluation.evaluate import evaluate_model_suite
from .reports.run_reports import write_all_reports


def run_residual_report(cfg: dict) -> dict:
    results = evaluate_model_suite(cfg)
    outdir = write_all_reports(cfg, results)
    c = results["claim_report"]
    print(f"[{c['endpoint_name']} / {c['endpoint_type']}] "
          f"technical_validation={c['technical_validation_claim_allowed']} "
          f"relapse_pfs={c['relapse_or_pfs_claim_allowed']} "
          f"omics={c['omics_increment_summary']} -> {outdir}")
    return {"outdir": str(outdir), "claim_report": c,
            "ablation": results["ablation"], "paired_deltas": results["paired_deltas"],
            "usefulness": results["usefulness"], "diag": results["diag"],
            "repeated_leaderboard": results["repeated_leaderboard"], "outcomes": results["outcomes"]}
