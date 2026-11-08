"""External-validation hook (P11).

Fit the ResidualRiskModel on the training cohort, then evaluate the FROZEN model
on an external cohort: held-out C-index, calibration, and an external claim
report. Builds the interface now so it is ready when a second cohort lands.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_config, ensure_outdir
from ..endpoints import resolve_endpoint
from ..models.residual_risk import ResidualRiskModel
from ..data.cohort import build_matched_cohort
from ..metrics import fast_c_index as cidx
from .stats import calibration_metrics
from .claim_gate import build_claim_report


def run_external_validation(train_config: str, external_config: str) -> dict:
    train_cfg = load_config(train_config)
    ext_cfg = load_config(external_config)
    schema = train_cfg["schema"]
    time_col, event_col = schema["time_col"], schema["event_col"]
    horizon = float(train_cfg.get("validation", {}).get("horizon_months", 24.0))
    outdir = ensure_outdir(ext_cfg["paths"].get("outdir", "outputs/external_validation"))

    train_df, groups = build_matched_cohort(train_cfg)
    ext_df, ext_groups = build_matched_cohort(ext_cfg)
    # use the intersection of feature groups present in both cohorts
    g = {k: [c for c in groups[k] if c in ext_df.columns] for k in groups}

    model = ResidualRiskModel(g["clinical"], g["cyto"] + g["omics"],
                              time_col=time_col, event_col=event_col).fit(train_df)
    pred = model.predict(ext_df)
    t = pd.to_numeric(ext_df[time_col]).values
    e = ext_df[event_col].astype(int).values
    c_total = cidx(t, e, pred.total_risk.values)
    c_clin = cidx(t, e, pred.clinical_risk.values)
    delta = c_total - c_clin
    cal, _ = calibration_metrics(t, e, 1 / (1 + np.exp(-(pred.total_risk.values - pred.total_risk.values.mean()))), horizon)

    endpoint = resolve_endpoint(ext_cfg)
    detail = {"external_n": int(len(ext_df)), "external_events": int(e.sum()),
              "clinical_cindex": round(float(c_clin), 4), "full_cindex": round(float(c_total), 4),
              "omics_paired_delta": round(float(delta), 4), "omics_paired_delta_ci_low": float("nan")}
    claim = build_claim_report(endpoint, omics_delta=delta, omics_delta_ci_low=float("nan"),
                               external_validation_available=True, detail=detail)

    pd.DataFrame([{"model": "clinical", "cindex": round(c_clin, 4)},
                  {"model": "total", "cindex": round(c_total, 4)},
                  {"model": "delta", "cindex": round(delta, 4)}]).to_csv(
        outdir / "external_calibration_metrics.csv", index=False)
    pd.DataFrame([cal]).to_csv(outdir / "external_calibration_summary.csv", index=False)
    (outdir / "external_claim_report.json").write_text(json.dumps(claim, indent=2))
    (outdir / "external_validation_report.md").write_text(
        f"# External validation — {endpoint['name']}\n\n"
        f"- External N={len(ext_df)}, events={int(e.sum())}\n"
        f"- Clinical C={c_clin:.3f}; total (clinical+molecular residual) C={c_total:.3f}; ΔC={delta:+.3f}\n"
        f"- External validation available: YES → omics increment {claim['omics_increment_summary']}\n"
        f"- evidence_level: {claim['evidence_level_for_omics_increment']}\n")
    print(f"External validation written -> {outdir}")
    return {"outdir": str(outdir), "claim": claim, "delta": delta}
