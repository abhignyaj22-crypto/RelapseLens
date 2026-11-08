"""Evaluation layer: C-index, paired ΔC, calibration, DCA, NRI/IDI, claim gating,
and the full evaluate_model_suite orchestrator."""
from ..metrics import fast_c_index, harrell_c_index, bootstrap_ci
from .stats import (paired_delta_cindex, evidence_level, calibration_metrics,
                    decision_curve, nri_idi)
from .claim_gate import build_claim_report
from .evaluate import evaluate_model_suite

__all__ = ["fast_c_index", "harrell_c_index", "bootstrap_ci", "paired_delta_cindex",
           "evidence_level", "calibration_metrics", "decision_curve", "nri_idi",
           "build_claim_report", "evaluate_model_suite"]
