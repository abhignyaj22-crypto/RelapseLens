"""Endpoint registry loader + claim gating.

The endpoint a run declares (cfg['endpoint']['name']) is resolved against
configs/endpoints.yaml. Claim flags are derived ONLY from the endpoint type and
the available evidence, so the code cannot emit a relapse/PFS claim on an OS run.
"""
from __future__ import annotations

from pathlib import Path
import yaml

RELAPSE_ENDPOINTS = {"pfs", "progression_free_survival", "time_to_progression",
                     "time_to_relapse", "early_progression", "early_progression_landmark"}

# Endpoints a clinical-anchored longitudinal claim may attach to (serial-clinical observed,
# calendar-time event). TT2L is included as a treatment-transition proxy; note it is NOT
# drug resistance (see resistance_language_lint in evaluation/claim_gate.py).
CLINICAL_ANCHORED_ENDPOINTS = RELAPSE_ENDPOINTS | {"ttnt_proxy", "time_to_next_treatment"}

# Minimum paired-molecular patients before a MOLECULAR longitudinal-trajectory claim is even
# considered. The open substrate has 29 — the molecular claim stays hard-blocked regardless
# of anything the clinical-anchored path earns.
MIN_MOLECULAR_PAIRS_FOR_TRAJECTORY = 100


def load_registry(path: str | Path = "configs/endpoints.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def resolve_endpoint(cfg: dict, registry_path: str | Path = "configs/endpoints.yaml") -> dict:
    """Return the endpoint spec for this run (name from cfg, type from registry)."""
    ep = cfg.get("endpoint", {}) or {}
    name = ep.get("name", "unknown")
    reg = load_registry(registry_path)
    spec = dict(reg.get(name, {}))
    spec.setdefault("endpoint_type", _infer_type(ep))
    spec["name"] = name
    spec["claim_scope"] = ep.get("claim_scope", spec.get("role", "unknown"))
    return spec


def _infer_type(ep: dict) -> str:
    scope = (ep.get("claim_scope") or "").lower()
    if "pfs" in scope or "progression" in scope:
        return "progression_free_survival"
    if "overall_survival" in scope or "os" in scope:
        return "overall_survival"
    return "overall_survival"


def gate_claims(endpoint_spec: dict, *, omics_increment_confirmed: bool,
                external_validation_available: bool, proposal_target: str = "relapse",
                clinical_serial_available: bool = False,
                beats_static_baselines_cindex: bool = False,
                calibration_win_over_kl: bool = False,
                is_synthetic: bool = False,
                n_molecular_pairs: int = 29) -> dict:
    """Compute the separated claim flags.

    technical_validation is always allowed once the pipeline runs on real data;
    the biological/relapse claims are hard-gated on endpoint type + evidence.

    Two longitudinal claims are kept DELIBERATELY distinct:
      * `clinically_anchored_longitudinal` — earnable now, but the SPLIT criterion reflects
        what the model is actually good at: it must beat the STATIC baselines (LOCF/Cox/RSF)
        on discrimination (C-index) AND beat Köhn-Luque on CALIBRATION (IBS). A smoothed-
        Köhn-Luque teacher ties KL on C-index by construction, so a C-index-only gate could
        never open; its edge is calibrated survival curves + honest responder handling.
      * `molecular_longitudinal_trajectory` — stays hard-blocked until ≥100 paired-molecular
        patients (the open substrate has 29). Nothing the clinical path earns unblocks it.
    """
    et = endpoint_spec.get("endpoint_type", "overall_survival")
    is_relapse_endpoint = et in RELAPSE_ENDPOINTS

    technical_validation_claim_allowed = True
    relapse_or_pfs_claim_allowed = bool(is_relapse_endpoint)
    primary_biological_claim_allowed = bool(
        is_relapse_endpoint and external_validation_available and omics_increment_confirmed
    )
    clinically_anchored_longitudinal_claim_allowed = bool(
        et in CLINICAL_ANCHORED_ENDPOINTS and clinical_serial_available
        and beats_static_baselines_cindex and calibration_win_over_kl and not is_synthetic
    )
    molecular_longitudinal_trajectory_claim_allowed = bool(
        n_molecular_pairs >= MIN_MOLECULAR_PAIRS_FOR_TRAJECTORY
    )
    return {
        "endpoint_name": endpoint_spec.get("name"),
        "endpoint_type": et,
        "endpoint_role": endpoint_spec.get("role", endpoint_spec.get("claim_scope")),
        "technical_validation_claim_allowed": technical_validation_claim_allowed,
        "primary_biological_claim_allowed": primary_biological_claim_allowed,
        "relapse_or_pfs_claim_allowed": relapse_or_pfs_claim_allowed,
        "clinically_anchored_longitudinal_claim_allowed": clinically_anchored_longitudinal_claim_allowed,
        "molecular_longitudinal_trajectory_claim_allowed": molecular_longitudinal_trajectory_claim_allowed,
        "beats_static_baselines_cindex": bool(beats_static_baselines_cindex),
        "calibration_win_over_kl": bool(calibration_win_over_kl),
        "n_molecular_pairs": int(n_molecular_pairs),
        "min_molecular_pairs_required": MIN_MOLECULAR_PAIRS_FOR_TRAJECTORY,
        "omics_increment_confirmed": bool(omics_increment_confirmed),
        "external_validation_available": bool(external_validation_available),
        "is_synthetic": bool(is_synthetic),
        "proposal_target": proposal_target,
    }
