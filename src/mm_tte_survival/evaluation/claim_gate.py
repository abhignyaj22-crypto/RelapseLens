"""Endpoint-gated claim computation (wraps endpoints.gate_claims + evidence_level)."""
from __future__ import annotations

import numpy as np

from ..endpoints import gate_claims
from .stats import evidence_level


def build_claim_report(endpoint_spec, *, omics_delta, omics_delta_ci_low,
                       external_validation_available, detail,
                       clinical_serial_available: bool = False,
                       beats_static_baselines_cindex: bool = False,
                       calibration_win_over_kl: bool = False,
                       is_synthetic: bool = False,
                       n_molecular_pairs: int = 29) -> dict:
    omics_confirmed = bool(np.isfinite(omics_delta_ci_low) and omics_delta_ci_low > 0)
    claims = gate_claims(endpoint_spec, omics_increment_confirmed=omics_confirmed,
                         external_validation_available=external_validation_available,
                         proposal_target="relapse",
                         clinical_serial_available=clinical_serial_available,
                         beats_static_baselines_cindex=beats_static_baselines_cindex,
                         calibration_win_over_kl=calibration_win_over_kl,
                         is_synthetic=is_synthetic,
                         n_molecular_pairs=n_molecular_pairs)
    ev = evidence_level(endpoint_spec.get("endpoint_type"), "relapse",
                        omics_delta, omics_delta_ci_low, external_validation_available)
    if np.isfinite(omics_delta) and omics_delta > 0 and not omics_confirmed:
        summary = "SUGGESTIVE_NOT_CONFIRMED"
    elif omics_confirmed:
        summary = "CONFIRMED"
    else:
        summary = "NO_EVIDENCE"
    return {**claims, "evidence_level_for_omics_increment": ev,
            "omics_increment_summary": summary, "_detail": detail}


# --------------------------------------------------------------------------- #
# Benchmark honesty linter — refuses "beats-SOTA" prose the evidence can't license
# --------------------------------------------------------------------------- #
SOTA_TERMS = ("beats sota", "beat sota", "beats state-of-the-art", "state-of-the-art",
              "state of the art", "outperform", "sota-beating", "best-in-class",
              "surpass", "new sota", "superior to sota")


RESISTANCE_TERMS = ("drug resistance", "drug-resistant", "treatment-resistant",
                    "resistance", "resistant", "refractory")


def resistance_language_lint(text: str, endpoint_spec: dict) -> dict:
    """Gate the word "resistance". It is licensed ONLY on a therapy-exposure-aware AND
    resistance-eligible endpoint. TT2L (time-to-next-treatment) is a treatment-transition
    proxy — exposure-aware but NOT resistance-eligible — so "resistance" prose on a TT2L run
    is a violation. Use progression/relapse/treatment-transition language instead.
    Returns {allowed, resistance_terms_used, therapy_exposure_aware, resistance_eligible,
    licensed, violations}.
    """
    lower = (text or "").lower()
    used = [t for t in RESISTANCE_TERMS if t in lower]
    exposure_aware = bool(endpoint_spec.get("therapy_exposure_aware"))
    resistance_eligible = bool(endpoint_spec.get("resistance_eligible"))
    licensed = exposure_aware and resistance_eligible
    allowed = (not used) or licensed
    return {"allowed": allowed, "resistance_terms_used": used,
            "therapy_exposure_aware": exposure_aware, "resistance_eligible": resistance_eligible,
            "licensed_for_resistance_language": licensed,
            "violations": [] if allowed else
            [f"resistance term(s) {used} not licensed on endpoint "
             f"'{endpoint_spec.get('name')}' (type={endpoint_spec.get('endpoint_type')}); "
             f"TT2L/proxy endpoints are treatment-transition, not resistance"]}


def benchmark_honesty_lint(text: str, evidence: dict) -> dict:
    """Gate superiority / SOTA-beating language. Such prose is licensed ONLY when the
    evidence is matched-cohort AND endpoint-matched AND ΔCI_low>0 AND FDR-significant
    AND externally validated. Any SOTA term used without all five is a violation.
    Returns {allowed, sota_terms_used, requirements, violations}.
    """
    lower = (text or "").lower()
    used = [t for t in SOTA_TERMS if t in lower]
    dl = evidence.get("delta_ci_low", float("nan"))
    req = {
        "matched_cohort": bool(evidence.get("matched_cohort")),
        "same_endpoint": bool(evidence.get("same_endpoint")),
        "delta_ci_low_positive": bool(np.isfinite(dl) and dl > 0),
        "fdr_significant": bool(evidence.get("fdr_significant")),
        "external_validation": bool(evidence.get("external_validation")),
    }
    licensed = all(req.values())
    allowed = (not used) or licensed
    missing = sorted(k for k, v in req.items() if not v)
    return {"allowed": allowed, "sota_terms_used": used, "requirements": req,
            "licensed_for_sota_language": licensed,
            "violations": [] if allowed else [f"SOTA term(s) {used} not licensed; missing {missing}"]}
