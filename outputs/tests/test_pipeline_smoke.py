from pathlib import Path
import yaml
import pandas as pd
from mm_tte_survival.demo import make_demo_data
from mm_tte_survival.audit import audit_inputs
from mm_tte_survival.experiments import run_experiments
from mm_tte_survival.residual import run_residual_report


def test_pipeline_smoke(tmp_path):
    data = tmp_path / "demo"
    make_demo_data(data, n=90, p=20, seed=7)
    report = audit_inputs(str(data / "clinical.csv"), str(data / "cytogenetics.csv"), str(data / "omics.csv"), tmp_path / "audit.json")
    assert report["usable_for"]["survival_tte"]
    cfg = {
        "seed": 7,
        "paths": {"clinical": str(data / "clinical.csv"), "cytogenetics": str(data / "cytogenetics.csv"), "omics": str(data / "omics.csv"), "outdir": str(tmp_path / "out")},
        "schema": {"id_col": "patient_id", "time_col": "time_months", "event_col": "event", "split_col": "split", "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "riss_2", "riss_3", "line_of_therapy"], "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "hyperdiploid"]},
        "features": {"use_clinical": True, "use_cytogenetics": True, "use_omics": True, "omics_pca_components": 5},
        "training": {"epochs": 6, "lr": 0.003, "hidden_dim": 16, "dropout": 0.05, "patience": 4, "distill_weight": 0.1, "ema_decay": 0.9, "temperature": 2.0},
        "experiments": {"models": ["cox", "opsd_cox"], "min_subtype_n": 5, "min_subtype_events": 2},
    }
    res = run_experiments(cfg)
    assert Path(res["outdir"], "leaderboard.csv").exists()
    assert len(res["leaderboard"]) == 2


def test_residual_report_smoke(tmp_path):
    data = tmp_path / "demo"
    make_demo_data(data, n=160, p=12, seed=11)
    cfg = {
        "seed": 11,
        "endpoint": {"name": "demo", "claim_scope": "demo_only"},
        "paths": {"clinical": str(data / "clinical.csv"), "cytogenetics": str(data / "cytogenetics.csv"),
                  "omics": str(data / "omics.csv"), "outdir": str(tmp_path / "out")},
        "schema": {"id_col": "patient_id", "time_col": "time_months", "event_col": "event", "split_col": "split",
                   "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy"],
                   "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "hyperdiploid"]},
        "cohort": {"require_omics": True, "require_cytogenetics": True, "matched_ablation": True},
        "validation": {"repeated_splits": 3, "horizon_months": 24},
        "experiments": {"min_test_events": 5, "min_subtype_events": 2,
                        "min_subtype_events_hypothesis": 5, "min_subtype_events_confirmatory": 15},
    }
    res = run_residual_report(cfg)
    out = Path(res["outdir"])
    # residual decomposition must be exactly additive
    d = pd.read_csv(out / "residual_risk_decomposition.csv")
    assert {"clinical_risk", "molecular_residual_risk", "total_risk"}.issubset(d.columns)
    assert ((d.clinical_risk + d.molecular_residual_risk) - d.total_risk).abs().max() < 1e-9
    # matched ablation compares all sets on the SAME patient count
    ab = pd.read_csv(out / "matched_ablation.csv")
    assert ab["n_patients"].nunique() == 1
    # separated, endpoint-gated claim keys present; OS-type endpoint cannot license relapse
    cr = res["claim_report"]
    for k in ["technical_validation_claim_allowed", "primary_biological_claim_allowed",
              "relapse_or_pfs_claim_allowed", "omics_increment_confirmed",
              "external_validation_available"]:
        assert k in cr
    assert cr["technical_validation_claim_allowed"] is True
    # new statistical artifacts written
    for f in ["paired_delta_cindex.csv", "repeated_split_leaderboard.csv",
              "calibration_metrics.csv", "decision_curve_analysis.csv",
              "mmrf_reclassification_outcomes.csv", "leakage_audit.json"]:
        assert (out / f).exists(), f
