"""CI guardrail: an OS-endpoint run must not emit relapse/PFS/clinical-use language."""
from pathlib import Path
import yaml
import pytest

from mm_tte_survival.demo import make_demo_data
from mm_tte_survival.main import run_pipeline

GUARDRAILS = Path("configs/reports/claim_language_guardrails.yaml")


def _forbidden_terms():
    if GUARDRAILS.exists():
        return [t.lower() for t in yaml.safe_load(GUARDRAILS.read_text())["forbidden_terms_if_endpoint_os"]]
    return ["relapse prediction", "pfs prediction", "resistance trajectory",
            "clinical decision support", "treatment recommendation"]


def test_os_run_has_no_forbidden_language(tmp_path):
    data = tmp_path / "demo"
    make_demo_data(data, n=160, p=12, seed=11)
    outdir = tmp_path / "out"
    cfg = {
        "seed": 11,
        "endpoint": {"name": "open_gdc_os"},
        "paths": {"clinical": str(data / "clinical.csv"), "cytogenetics": str(data / "cytogenetics.csv"),
                  "omics": str(data / "omics.csv"), "outdir": str(outdir)},
        "schema": {"id_col": "patient_id", "time_col": "time_months", "event_col": "event",
                   "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy"],
                   "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "hyperdiploid"]},
        "cohort": {"matched_ablation": True},
        "evaluation": {"repeated_splits": 3, "horizon_months": 24, "min_test_events": 5,
                       "min_subtype_events_hypothesis": 5, "min_subtype_events_confirmatory": 15},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    run_pipeline(str(tmp_path / "cfg.yaml"))

    forbidden = _forbidden_terms()
    offenders = []
    for md in outdir.glob("*.md"):
        text = md.read_text().lower()
        for term in forbidden:
            if term in text:
                offenders.append((md.name, term))
    assert not offenders, f"forbidden clinical-claim language on an OS run: {offenders}"


def test_os_claim_gate_blocks_relapse(tmp_path):
    from mm_tte_survival.endpoints import resolve_endpoint, gate_claims
    spec = resolve_endpoint({"endpoint": {"name": "open_gdc_os"}})
    claims = gate_claims(spec, omics_increment_confirmed=True,
                         external_validation_available=True, proposal_target="relapse")
    # even with confirmed omics + external validation, OS cannot license relapse/PFS
    assert claims["relapse_or_pfs_claim_allowed"] is False
    assert claims["primary_biological_claim_allowed"] is False
    assert claims["technical_validation_claim_allowed"] is True
